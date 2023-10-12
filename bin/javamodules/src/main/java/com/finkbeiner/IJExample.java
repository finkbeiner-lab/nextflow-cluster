package com.finkbeiner;

import ij.ImagePlus;
import ij.process.ImageProcessor;
import ij.process.ImageConverter;
import ij.io.Opener;

public class IJExample {
    public static void main(String[] args) {
        // Load an image using ImageJ
        Opener opener = new Opener();
        ImagePlus imagePlus = opener.openImage("/workspace/finkbeiner-profile.jpg");

        // Convert the image to grayscale
        ImageConverter converter = new ImageConverter(imagePlus);
        converter.convertToGray8();

        // Get the image processor
        ImageProcessor processor = imagePlus.getProcessor();

        // Perform some image processing operations (e.g., invert colors)
        processor.invert();

        // Save the processed image
        ij.IJ.save(imagePlus, "/gladstone/finkbeiner/lab/finkbeiner-profile-grayscale.jpg");

    }
}

