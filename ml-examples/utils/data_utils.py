import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

def delete_data_subdirs(loc) :
    directory = Path(loc)
    for item in directory.iterdir():
        if item.is_dir():
            delete_data_subdirs(item)
        else:
            item.unlink()

def vnorm(im) :
    return np.sqrt(np.power(100.*(im[:,:,0] - 127.5)/255.,2) + np.power(100.*(im[:,:,1] - 127.5)/255.,2))

def generate_data_from_datasets(dest, loc, datasets, curlThreshold = 0.49, velocityThreshold = 18.0) :
    """
    Function to move images from datasets/ to data/ with the appropriate file structure.
    
    Arguments:
    dest -- where to put data (e.g. data/)
    loc -- where the datasets are located (e.g. datasets/)
    datasets -- a triple of dictionaries of the form "directory_name" : number_images_to_extract.
                Should be ordered as (train, dev, test)
    
    If number_images_to_extract is larger than the size of the directory, 
    or number_images_to_extract == "*", this function will collect the 
    maximum number available.

    Returns: None
    """
    
    # Get a list of filename elements that correspond to truth-level labels
    vecOfExts=["v","d","c","p"]
    
    for segment, sets in datasets.items(): 
        # Make destination directory structure in case it does not exist
        os.makedirs(dest+"/"+segment+"/image/img", exist_ok=True)
        os.makedirs(dest+"/"+segment+"/mask/img", exist_ok=True)
        
        for dataset, numToTake in sets.items() :
            
            # Announce dataset being collected
            print("For segment",segment,", processing",numToTake,"images from dataset",dataset)
            
            # Get the number of images in each of the image and mask directories
            numIms = np.max([0]+[int(tFile.split(".")[0] if tFile != ".ipynb_checkpoints" else 0) for tFile in os.listdir(dest+"/"+segment+"/image/img/")])
            numMsk = np.max([0]+[int(tFile.split(".")[0] if tFile != ".ipynb_checkpoints" else 0) for tFile in os.listdir(dest+"/"+segment+"/mask/img/")])
            
            # Report findings
            assert numIms == numMsk, "\t - ERROR: Number of images does not equal number of masks. Please fix.";
            print("\t - Found",numIms,"images in",segment,"image directory")
            
            # Initialize helper variables
            numImAdded=0
            datasetDir=loc+"/"+dataset+"/"
            
            # Announce intentions
            numImAvailable=len([dName for dName in os.listdir(datasetDir) if (dName.split('_'))[-2] not in vecOfExts])
            if(numToTake == "*") :
                numToTake = numImAvailable
            print("\t - Pulling ",str(numToTake),"images from",datasetDir,"out of max",numImAvailable)
            
            # Loop over dye images in the dataset directory
            for dName in os.listdir(datasetDir):
            
                # Stop adding files at the specified number
                if(numImAdded >= numToTake): 
                    print(" - Done with dataset \n")
                    break;
            
                # Only process snapshots of the simulation itself
                if((dName.split('_'))[-2] in vecOfExts): continue;
            
                # Get the simulation index
                idx = int((dName).split("_")[-1].split(".")[0])
                #print("\t\t - Processing image ",idx)
            
                # Load the relevant images for Input (image) and Output (mask)
                dIm, vIm, cIm = None,None,None
                try: 
                    # Get names of velocity and curl images from the principal name
                    vName = '_'.join(dName.split("_")[:-1]+["v",str(idx)+".png"])
                    cName = '_'.join(dName.split("_")[:-1]+["c",str(idx)+".png"])
                    
                    # Load helper images for processing output
                    #print("\t\t - Loading images")
                    dIm = np.array(Image.open(datasetDir+"/"+dName))[:,:,:-1]
                    vIm = np.array(Image.open(datasetDir+"/"+vName))[:,:,:-1]
                    cIm = np.array(Image.open(datasetDir+"/"+cName))[:,:,:-1]
                except OSError:
                    print("\t\t - ERROR: Could not open files at "+ datasetDir +", names "+ dName +", "+ vName +", "+ cName +".")
                    continue;
                
                
                # Get the save index
                numIms += 1
                newIdx = numIms
            
                # Compute the mask
                msk = (np.abs(cIm[:,:,0] - 127.5)/255. >= curlThreshold) * (vnorm(vIm) >= velocityThreshold)
                #print("\t\t - Image shapes are",dIm.shape, vIm.shape, cIm.shape, msk.shape)
        
                #if(numImAdded % 50 == 0) :
                #print("\t\t - Saving",dName,"to image/img folder as",str(newIdx)+".png")
                #print("\t\t - Saving mask to mask/img folder as",str(newIdx)+".png")
        
                # Save file
                Image.fromarray(dIm).save(dest+"/"+segment+"/image/img/"+str(newIdx)+".png")
                Image.fromarray(msk).save(dest+"/"+segment+"/mask/img/"+str(newIdx)+".png")
            
                numImAdded+=1