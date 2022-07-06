import numpy as np
import cv2
import imutils
import os
import time
import matplotlib.pyplot as plt
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

if __name__ == "__main__":

    while True:

        print("\n")
        print("CHOOSE YOUR OPTION: ")
        print("\n")
        print("1. FACE-MASK DETECTION")
        print("2. CROWD DETECTION")    
        print("3. EXIT")
        print("\n")

        choice=int(input("Enter your choice: "))
        print("\n")

        if choice==1:
            os.system('python test.py')
        
        elif choice==2:
            os.system('python test2.py')

        elif choice==3:
            break

        else:
            print("Wrong Choice")