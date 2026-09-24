#!/usr/bin/env python3
"""Decode every frame and produce a full-duration contact sheet for review."""
import sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from mjlab_microduck.jump_artifacts import atomic_json
video,output=sys.argv[1:]
reader=imageio.get_reader(video);meta=reader.get_meta_data();count=0;frames=[];bad=0
for index,frame in enumerate(reader):
    count+=1
    if frame.shape!=(480,640,3) or frame.std()<1.:bad+=1
    if index%50==0:frames.append((index,frame.copy()))
reader.close()
sheet=Image.new('RGB',(640*5,500*((len(frames)+4)//5)))
for j,(index,frame) in enumerate(frames):
    x=j%5*640;y=j//5*500
    sheet.paste(Image.fromarray(frame),(x,y));ImageDraw.Draw(sheet).text((x+5,y+481),f'{index/50:.1f} seconds',fill='white')
sheet.save(Path(output).with_suffix('.jpg'))
passed=count>=1500 and abs(meta['fps']-50)<.01 and bad==0
atomic_json(output,{'passed':passed,'decoded_frames':count,'fps':meta['fps'],'seconds':count/meta['fps'],
    'invalid_frames':bad,'contact_sheet':str(Path(output).with_suffix('.jpg')),
    'human_visual_review':'Contact sheet generated; automated decode is not human behavior acceptance.'})
raise SystemExit(0 if passed else 1)
