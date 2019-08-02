import os, sys
import copy
import pickle
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

BG_SIZE = 512

def resize_bg(src_dir, dst_dir):
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    
    for fname in os.listdir(src_dir):
        img = Image.open(os.path.join(src_dir, fname))
        # crop and resize
        x, y = img.size
        sz = min(x, y)
        box = (0, 0, sz, sz)
        new_bg = img.crop(box)
        new_bg = img.resize((BG_SIZE, BG_SIZE), Image.ANTIALIAS)
        # save image
        new_fname = os.path.join(dst_dir, fname)
        new_bg.save(new_fname)

def generate_list(root):
    all_files = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(root)] for val in sublist]
    return all_files

def read_list(fname):
    with open(fname) as f:
        data = f.read().strip().split()
        return data

def rotation(degree, img):
    return img.rotate(degree,fillcolor='white', expand=True)

def sample_sen(lst, bg, degree):
    bg_sz = bg.size
    sen_w = bg.size[0] // 2
    T = 1500
    x, y = 0, 0
    while x < 1500 or x > 1900:
        idx = random.randint(0, len(lst)-1)
        img = Image.open(lst[idx])
        x, y = img.size
    scale = max(1, img.size[0] // sen_w)
    return binarize(rotation(degree, img.resize((x//scale, y//scale))))
 
def sample_word(lst, bg, degree):
    bg_sz = bg.size
    sen_w = bg.size[0] // 11
    T = 50
    x, y = 0, 0
    while x < 50 or x > 90:
        idx = random.randint(0, len(lst)-1)
        img = Image.open(lst[idx])
        x, y = img.size
    scale = img.size[0] / sen_w
    return binarize(rotation(degree, img.resize((int(x//scale), int(y//scale)))))
    
def binarize(img):
    T = 220
    img_np = np.array(img)
    img_np[img_np>T] = 255
    return Image.fromarray(img_np)
    
def mask_paste(img, sub, x, y):
    pixels = list(sub.getdata())
    for index, colour in enumerate(pixels):
        if not colour == 255:
            img.putpixel((index%sub.size[0] + x, index//sub.size[0] + y), colour)
    return img

def overlap(box1, box2):
    #print (f'OverLap ---> sen : {box2}, word : {box1}')
    L1x, L1y, R1x, R1y = box1
    L2x, L2y, R2x, R2y = box2
    if (L1x >= R2x or L2x >= R1x):
        return False
    if (L1y >= R2y or L2y >= R1y):
        return False
    return True 
    
def paste_img(sen, word, img):
    x, y = img.size
    #print ("img size = ", img.size)
    #input()
    while True:
        x2, y2 = random.randint(0, x-sen.size[0]), random.randint(0, y-sen.size[1])
        #x2, y2 = random.randint(0, x-200), random.randint(0, y)
        #print (f'sen : ({x2}, {y2}), size = {sen.size}')
        x1, y1 = x2, y2
        while overlap((x1, y1, x1+word.size[0], y1+word.size[1]), (x2, y2, x2+sen.size[0], y2+sen.size[1])):
            x1, y1 = random.randint(0, x-word.size[0]), random.randint(0, y-word.size[1])
            #x1, y1 = random.randint(0, x-200), random.randint(0, y//2)
            #print (f'word : ({x1}, {y1}), size = {word.size}')
            #input()
        if not (x2+sen.size[0] > x or y2+sen.size[1] > y or x1+word.size[0] > x or y1+word.size[1] > y):
            #if img.getpixel((x1, y1)) > 200 and img.getpixel((x2, y2)) > 200:
            break
    #print ("Out")
    # generate files
    original = mask_paste(img, word, x1, y1)
    original = mask_paste(img, sen, x2, y2)
    white_bg = Image.new('L', img.size, 255)
    handwritten = mask_paste(white_bg, word, x1, y1)
    handwritten = mask_paste(white_bg, sen, x2, y2)

    word_box = (x1, y1, x1+word.size[0], y1+word.size[1])
    sen_box = (x2, y2, x2+sen.size[0], y2+sen.size[1])
    anno = {'word':[word_box], 'sen':[sen_box]}

    return original, white_bg, anno

def re_scale(original, handwriting, tmp_printed):
    #target_sz = (960, 1280)
    X, Y = 480, 640
    target_sz = (X, Y)
    input_sz = original.size

    printed = copy.deepcopy(tmp_printed)

    if input_sz[0] > input_sz[1]:
        printed = printed.transpose(Image.ROTATE_90)
        original = original.transpose(Image.ROTATE_90)
        handwriting = handwriting.transpose(Image.ROTATE_90)


    new_original = Image.new('L', target_sz, 255)
    new_printed = Image.new('L', target_sz, 255)
    new_handwriting = Image.new('L', target_sz, 255)

    if input_sz[0] / input_sz[1] <= 0.75:
        new_sz = (int(input_sz[0]/input_sz[1]*Y), Y)
        original = original.resize(new_sz)
        printed = printed.resize(new_sz)
        handwriting = handwriting.resize(new_sz)

        new_original.paste(original, ((X-new_sz[0])//2, 0))
        new_printed.paste(printed, ((X-new_sz[0])//2, 0))
        new_handwriting.paste(handwriting, ((X-new_sz[0])//2, 0))
    else:
        new_sz = (X, int(input_sz[1]/input_sz[0]*X))
        original = original.resize(new_sz)
        printed = printed.resize(new_sz)
        handwriting = handwriting.resize(new_sz)

        new_original.paste(original, (0, (Y-new_sz[1])//2))
        new_printed.paste(printed, (0, (Y-new_sz[1])//2))
        new_handwriting.paste(handwriting, (0, (Y-new_sz[1])//2) )

    assert new_original.size == target_sz

    return new_original, new_handwriting, new_printed

if __name__ == "__main__":
    src_dir = "english_form"
    dst_dir = "XDD"
    # destination
    #handwriting_dst = "Signature-detection/data/handwriting"
    #annotation_dst = "Signature-detection/data/anns"
    #original_dst = "Signature-detection/data/original"
    #printed_dst = "Signature-detection/data/printed"


    handwriting_dst = "Signature-detection/test/handwriting"
    annotation_dst = "Signature-detection/test/anns"
    original_dst = "Signature-detection/test/original"
    printed_dst = "Signature-detection/test/printed"
 

    # load file lists
    sentence_files = read_list('sentence_list.txt')
    num_of_sentence = len(sentence_files)
    word_files = read_list('word_list.txt')
    num_of_word = len(word_files)
    annotation = {}
    #data = os.listdir(src_dir)     
    count = 0


    data = read_list('data_list.txt')     
    for fname in tqdm(data):
        form = Image.open(os.path.join(fname))
        for i in range(20):
            img_tmp = copy.deepcopy(form)
            img_tmp_2 = copy.deepcopy(form)
            # generate word and sentence
            word_degree = random.uniform(-10, 10)
            sen_degree = random.uniform(-10, 10)
            sen = sample_sen(sentence_files, img_tmp, sen_degree)
            word = sample_word(word_files, img_tmp, word_degree)
            # generate original, handwriting, annotation
            original, handwriting, anno = paste_img(sen, word, img_tmp)
            #print (original.size)
            #continue
            original, handwriting, printed = re_scale(original, handwriting, img_tmp_2)
            # save to file
            newfname = f'data_{count}.png'
            original.save(os.path.join(original_dst, newfname))
            handwriting.save(os.path.join(handwriting_dst, newfname))
            printed.save(os.path.join(printed_dst, newfname))
            annotation[newfname] = anno
            count += 1
#    with open(os.path.join(annotation_dst, "data.pkl"), "wb") as f:
#        pickle.dump(annotation, f)
#resize_bg(src_dir, dst_dir)
#word_root = '/var/ctc6nlp/datarecibo/iam_dataset/www.fki.inf.unibe.ch/DBs/iamDB/data/words/'   
#sentence_root = '/var/ctc6nlp/datarecibo/iam_dataset/www.fki.inf.unibe.ch/DBs/iamDB/data/sentences/'   
#word_files = generate_list(word_root)
#sentence_files = generate_list(sentence_root)

#word_files = read_list('word_list.txt')


