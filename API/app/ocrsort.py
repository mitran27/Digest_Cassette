# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:52:57 2021

@author: mitran
"""
import numpy as np
import cv2
import math

def ptn4box(*pts):
    return int(sum(pts)/len(pts))

def taninv(x):
    return math.degrees(math.atan(x))

def dist(pt1,pt2):
   
    return math.sqrt(abs(pt1[0]-pt2[0])**2+abs(pt1[1]-pt2[1])**2)


def rotate(origin, point, angle):
  
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)
def ocrsort(data):
    
        
        image=np.zeros((2000,2000,3),dtype=np.uint8)
        origin=[1000,1000]
        
        
        angle=0
        val=0
        
        
        for ids,record in enumerate(data,start=1):
            
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], word,_ =record
            
            
            
            cl=((x1+x4)//2,(y1+y4)//2)
            cr=((x2+x3)//2,(y2+y3)//2)
            height = abs(cr[1]-cl[1])
            width  = abs(cr[0]-cl[0])
            angle+=taninv(height/width)*len(word)
            val+=len(word)            
            
          
        print(math.ceil(angle/val))
        
        rotangle=math.radians(math.ceil(angle/val)+1)
        
        rowarr=[False]*2000
        
        for ids,record in enumerate(data,start=1):
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], word,_ =record
            
            cx=ptn4box(x1,x2,x3,x4)
            cy=ptn4box(y1,y2,y3,y4)
            
            cx,cy=rotate(origin, (cx, cy),rotangle)
            
            cv2.circle(image,(cx,cy),3,(100,100,255),-1)
            
            l=dist([x1, y1],[x4, y4])
            r=dist([x2, y2],[x3, y3])
            
            height=min(l,r)/2
            
            up=int(cy-height//2) 
            down=int(cy+height//2 +1)
            
            image = cv2.line(image,(cx,up) , (cx,down), (255,45,0), 10)

            if(type(rowarr[up])==bool):
                rowarr[up]=[]     
                
            #record.append((cx,cy))
            
            rowarr[up].append([word,(cx,cy)])
            for i in range(up+1,down):
                if(rowarr[i]==False):rowarr[i]=True
                
        
        paragraph=[]
        line=[]
        for i in range(2000):
            if(rowarr[i]==False):
                if(len(line)):
                    line=sorted(line,key=lambda x: x[-1][0])
                    paragraph.append(line)
                    line=[]
                
            else:
                if(type(rowarr[i])==list):
                    line.extend(rowarr[i])
        return paragraph
        
                
                
                
                
            
            
            
            
            
            
            
            #cv2.putText(image,  text = word,  org = (cx, cy),  fontFace = cv2.FONT_HERSHEY_DUPLEX,  fontScale = 0.5,  color = (125, 246, 55),  thickness = 1)
            
            
            
        
        
        
        
        
        cv2.imwrite("asd.jpeg",image) 
    