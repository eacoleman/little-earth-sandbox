import os
import sys
import numpy as np

def vnorm(im) :
    return np.sqrt(np.power(100.*(im[:,:,0] - 0.5),2) + np.power(100.*(im[:,:,1] - 0.5),2))

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

    # Make destination directory structure in case it does not exist
    os.mkdir(dest+"/image")
    os.mkdir(dest+"/mask")
    os.mkdir(dest+"/image/img")
    os.mkdir(dest+"/mask/img")
    
    # Get the number of images in each of the image and mask directories
    numIms = np.max([int(tFile.split(".")[0]) for tFile in os.listdir(dest+"/image/img/")])
    numMsk = np.max([int(tFile.split(".")[0]) for tFile in os.listdir(dest+"/mask/img/")])
    
    assert(numIms == numMsk, "ERROR: Number of images does not equal number of masks. Please fix.")

    for segment, sets in datasets.items(): 
        for dataset, numToTake in sets.items() :
            numImAdded=0
            datasetDir=loc+"/"+dataset+"/"
            
            print("Pulling " + str(numToTake) + " images from " + datasetDir)
            
            # Loop over dye images in the dataset directory
            for dName in os.listdir(datasetDir):
            
                # Stop adding files at the specified number
                if(numImAdded != "*" && numImAdded > numToTake) break;
            
                # Only process snapshots of the simulation itself
                if(dName.split[-2] in vecOfExts) continue;
            
                # Get the simulation index
                idx = int((dName).split("_")[-1].split(".")[0])
            
                # Load the relevant images for Input (image) and Output (mask)
                dIm, vIm, cIm = None,None,None
                try: 
                    vName = '_'.join(dName.split("_")[:-2]+["v",str(idx)+".png"])
                    cName = '_'.join(dName.split("_")[:-2]+["c",str(idx)+".png"])
                
                    dIm = np.array(Image.open(datasetDir+"/"+dName))[:,:,:-1]
                    vIm = np.array(Image.open(datasetDir+"/"+vName))[:,:,:-1]
                    cIm = np.array(Image.open(datasetDir+"/"+cName))[:,:,:-1]
                except OSError:
                    print("Could not open files at "+ loc +", names "+ dName +", "+ vName +", "+ cName +".")
                
                # Get the save index
                numIms += 1
                newIdx = numIms
            
                # Compute the mask
                msk = (np.abs(cIm[:,:,0] - 0.5) >= curlThreshold) * (vnorm(vIm) >= velocityThreshold)
        
                if(numImAdded % 50 == 0) :
                    print("\t - For file " + str(newIdx) + ".png")
                    print("\t\t - Saving " + dName + " to image/img folder.")
                    print("\t\t - Saving mask to mask/img folder.")
        
                # Save file
                Image.fromarray(dIm).save(dest+"/"+segment+"/image/img/"+str(newIdx)+".png")
                Image.fromarray(msk).save(dest+"/"+segment+"/mask/img/"+str(newIdx)+".png")
            
                numImAdded+=1