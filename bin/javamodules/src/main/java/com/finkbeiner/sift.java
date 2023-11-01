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

        String parent_dir = "/gladstone/finkbeiner/robodata/Robo4Images/20230928-MsNeu-RGEDItau1/A8/";
        String src_file = parent_dir + "PID20230928_20230928-MsNeu-RGEDItau1_T0_0.0-0_A8_1_Confocal-GFP16_0_0_1.tif";
        String dst_file = parent_dir + "PID20230929_20230928-MsNeu-RGEDItau1_T1_12.0-0_A8_1_Confocal-GFP16_0_0_1.tif";
        String outputStackPath = "registered.tif";
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
        SIFT()
        ij.IJ.run("Linear Stack Alignment with SIFT...", 
        "initial_gaussian_blur=1.60 steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=4 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Rigid interpolate");

        // Show the aligned stack
        // alignedStack.show();

        // Optionally, save the aligned stack to a file
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

