import sys
import numpy as np 
from statistics import median
from PIL import Image
import matplotlib.pyplot as plt


#
def MultipleStitch(Images, Trans, fileName):
    '''
    MultipleStitch 
    This function stitches multiple Images together and outputs the panoramic stitched image
    with a chain of input Images and its corresponding Transformations. 
    
    Given a chain of Images:
        I1 -> I2 -> I3 -> ... -> Im
    and its corresponding Transformations:
        T1 Transforms I1 to I2
        T2 Transforms I2 to I3
        ....
        Tm-1 Transforms Im-1 to Im
    
    The middle image was chosen as the reference image, and the outputed
    panorama is in the same coordinate system as the reference image.  
    Originally, we have
        I1 -> I2 -> ... -> Iref -> ... -> Im-1 -> Im
    When we fix Iref as the final coordinate system, we want all other
    Images Transformed to Iref.
        
    INPUTS:
    - Images: m list, each cell contains an image
    - Trans: (m-1) list, each element i contains an affine Transformation matrix that Transforms i to i+1.
    - fileName: the output file name.
        
    OUTPUTS:
    - Pano: the final panoramic image.
    '''
    
    # Check input formats
    if len(Images) != len(Trans)+1:
        sys.exit('Number of Images does not match the number of Transformations.')

    # Outbounds of panorama image
    outBounds = np.zeros([2,2])
    outBounds[0,:] = np.Inf
    outBounds[1,:] = -np.Inf

    # Choose reference image Iref
    # refIdx = np.floor(median(range(len(Images)))).astype('int')
    refIdx = 3

    # Estimate the largest possible panorama size
    [ncols, nrows] = next(iter(Images.values())).size
    nrows = len(Images)*nrows
    ncols = len(Images)*ncols
    
    # imgToRefTrans is a list of length m where imgToRefTrans[i] gives the
    # affine Transformation from Images[i] to the reference image Images[refIdx].
    imgToRefTrans = []
    
    # Initialize imgToRefTrans to contain the identity Transform.
    for _ in range(len(Images)):
        imgToRefTrans.append(np.eye(3))

    # Find the correct Transformations to reference Images and estimate possible output bounds
    for idx in range(len(Images)):
        imgToRefTrans[idx] = makeTransformToReferenceFrame(Trans, idx, refIdx)
        tmpBounds = findAffineBound(Images[idx], imgToRefTrans[idx])
        outBounds[0,:] = np.minimum(outBounds[0,:], tmpBounds[0,:])
        outBounds[1,:] = np.maximum(outBounds[1,:],tmpBounds[1,:])
    
    # Stitch the Iref image.
    H = np.eye(3)
    Pano = affineTransform(Images[refIdx], H, outBounds, nrows, ncols)

    # Transform the Images from the left side of Iref using the correct Transformations you computed
    for idx in range(refIdx-1,-1,-1):
        T = imgToRefTrans[idx]
        AddOn = affineTransform(Images[idx], T, outBounds, nrows, ncols)
        result_mask = np.sum(Pano, axis=2) != 0
        temp_mask = np.sum(AddOn, axis=2) != 0
        add_mask = temp_mask & (~result_mask)
        for c in range(Pano.shape[2]):
            cur_im = Pano[:,:,c]
            temp_im = AddOn[:,:,c]
            cur_im[add_mask] = temp_im[add_mask]
            Pano[:,:,c] = cur_im
        

    
    # Transform the Images from the right side of Iref using the correct Transformations you computed
    for idx in range(refIdx+1,len(Images)):
        T = imgToRefTrans[idx]
        AddOn = affineTransform(Images[idx], T, outBounds, nrows, ncols)
        result_mask = np.sum(Pano, axis=2) != 0
        temp_mask   = np.sum(AddOn, axis=2) != 0
        add_mask    = temp_mask & (~result_mask)
        for c in range(Pano.shape[2]):
            cur_im           = Pano[:,:,c]
            temp_im          = AddOn[:,:,c]
            cur_im[add_mask] = temp_im[add_mask]
            Pano[:,:,c]      = cur_im

    # Cropping the final panorama to leave out black spaces.
    boundMask = np.where(np.sum(Pano, axis=2) != 0)
    Pano = Pano[min(boundMask[0]):max(boundMask[0]),min(boundMask[1]):max(boundMask[1])]

    return Pano 


#
def makeTransformToReferenceFrame(i_To_iPlusOne_Transform, currentFrameIndex, refFrameIndex):
    '''
    makeTransformToReferenceFrame
    INPUT:
    - i_To_iPlusOne_Transform: this is a list contains i_To_iPlusOne_Transform[i] 
        contains the 3x3 homogeneous Transformation matrix that Transforms a point in frame 
        i to the corresponding point in frame i+1
    
    - currentFrameIndex: index of the current coordinate frame in i_To_iPlusOne_Transform
    - refFrameIndex: index of the reference coordinate frame

    OUTPUT:
    - T: A 3x3 homogeneous Transformation matrix that would convert a point in the current frame into the 
        corresponding point in the reference frame. For example, if the current frame is 2 and the reference frame 
        is 3, then T = i_To_iPlusOne_Transform{2}. If the current frame and reference frame are not adjacent, 
        T will need to be calculated.
    '''
    
    #############################################################################
    #                                                                           #
    # Sample batch_size elements from the training data and their               #
    # corresponding labels to use in this round of gradient descent.            #
    # Store the data in X_batch and their corresponding labels in               #
    # y_batch; after sampling X_batch should have shape (dim, batch_size)       #
    # and y_batch should have shape (batch_size,)                               #
    #                                                                           #
    #############################################################################        
    
    T = np.identity(3)
    if currentFrameIndex<refFrameIndex:
        for i in range(currentFrameIndex,refFrameIndex):
            T = i_To_iPlusOne_Transform[i]@T
            
    else:
        for i in range(refFrameIndex,currentFrameIndex):
            T = i_To_iPlusOne_Transform[i]@T
        T = np.linalg.pinv(T)   
    
    return T    


# find the output boundaries after affine transform
def findAffineBound(img, H):
    yLength, xLength, _ = np.asarray(img).shape
    urPoint = np.asarray([[yLength, yLength, 1]])
    ulPoint = np.asarray([[0, yLength, 1]])
    brPoint = np.asarray([[yLength, 0, 1]])
    blPoint = np.asarray([[0, 0, 1]])
    
    urAffine = np.dot(urPoint, H.T)
    ulAffine = np.dot(ulPoint, H.T)
    brAffine = np.dot(brPoint, H.T)
    blAffine = np.dot(blPoint, H.T)

    xMax = max(urAffine[0,0], ulAffine[0,0], brAffine[0,0], blAffine[0,0])
    yMax = max(urAffine[0,1], ulAffine[0,1], brAffine[0,1], blAffine[0,1])
    xMin = min(urAffine[0,0], ulAffine[0,0], brAffine[0,0], blAffine[0,0])
    yMin = min(urAffine[0,1], ulAffine[0,1], brAffine[0,1], blAffine[0,1])
    tmpBounds = np.asarray([[xMin, yMin], [xMax, yMax]])

    return tmpBounds


# This function perform affine transform with given output boundaries and size
def affineTransform(img, H, outBounds, nrows, ncols):
    tmp     = np.asarray(img)
    channel = tmp.shape[2]
    minX    = int(outBounds[0,0])
    minY    = int(outBounds[0,1])

    if minY < 0:            
        img = np.zeros([tmp.shape[0]-minY, tmp.shape[1], channel]).astype('uint8')
        img[-minY:,:,:] = tmp
    if minX < 0:            
        img = np.zeros([tmp.shape[0], tmp.shape[1]-minX, channel]).astype('uint8')
        img[:,-minX:,:] = tmp
    
    Hinv      = np.linalg.inv(H)
    Hinvtuple = (Hinv[0,0],Hinv[0,1], Hinv[0,2], Hinv[1,0],Hinv[1,1],Hinv[1,2])
    affine    = np.array(Image.fromarray(img).transform((ncols,nrows),  Image.AFFINE, Hinvtuple))

    return affine
