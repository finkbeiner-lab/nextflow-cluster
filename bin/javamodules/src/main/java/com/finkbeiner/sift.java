package com.finkbeiner;

import ij.ImagePlus;
import ij.process.ImageProcessor;
import ij.process.ImageConverter;
import ij.ImageStack;
import java.io.File;
import mpicbg.ij.SIFT;
import ij.io.Opener;
import ij.IJ;
import ij.plugin.frame.RoiManager;


public class sift {
    
    public static void main(String[] args) {


        String src_file = "PID20230928_20230928-MsNeu-RGEDItau1_T0_0.0-0_A8_1_Confocal-GFP16_0_0_1.tif";
        String dst_file = "PID20230929_20230928-MsNeu-RGEDItau1_T1_12.0-0_A8_1_Confocal-GFP16_0_0_1.tif";
        ImageStack stack = new ImageStack(1, 1);
        ImagePlus src_tile = IJ.openImage(src_file);

        
        // Add each image to the stack
        stack.addSlice(src_tile.getProcessor());
        // Optionally, set the title for each slice (image)
        stack.setSliceLabel(src_tile.getShortTitle(), stack.getSize());
        src_tile.close();

        ImagePlus dst_tile = IJ.openImage(dst_file);
        // Add each image to the stack
        stack.addSlice(dst_tile.getProcessor());
        // Optionally, set the title for each slice (image)
        stack.setSliceLabel(dst_tile.getShortTitle(), stack.getSize());
        dst_tile.close();
        
        // Create an ImagePlus with the stacked images
        ImagePlus imp = new ImagePlus("Stacked Images", stack);
        // Replace with the path to your input stack of images

        // ImageStack stack = imp.getStack();
        
        // Create a new ImagePlus for the aligned stack
        ImagePlus alignedStack = new ImagePlus("Aligned Stack", stack);

        // Initialize SIFT parameters
        SIFT sift = SIFT.matchFeatures(null, null, null, 0);
        SIFT sift = new SIFT(); 
        sift.setOptions("action=Align channels=1-2 reference=1 display_images save_transform");
        sift.run(alignedStack);

        // Show the aligned stack
        alignedStack.show();

        // Optionally, save the aligned stack to a file
        String outputStackPath = "registered.tif";
        IJ.save(alignedStack, outputStackPath);

        // Close the original stack and the aligned stack
        imp.close();
        alignedStack.close();
        
        // Optional: Export the transformation parameters
        RoiManager roiManager = RoiManager.getInstance();
        if (roiManager != null) {
            roiManager.runCommand("Save", "path_to_transform_parameters.txt");
        }
    }
}

