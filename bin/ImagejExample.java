import ij.ImagePlus;
import ij.process.ImageProcessor;
import ij.process.ImageConverter;
import ij.io.Opener;

public class ImageJExample {
    public static void main(String[] args) {
        // Load an image using ImageJ
        Opener opener = new Opener();
        ImagePlus imagePlus = opener.openImage("path/to/your/image.jpg");

        // Convert the image to grayscale
        ImageConverter converter = new ImageConverter(imagePlus);
        converter.convertToGray8();

        // Get the image processor
        ImageProcessor processor = imagePlus.getProcessor();

        // Perform some image processing operations (e.g., invert colors)
        processor.invert();

        // Save the processed image
        imagePlus.saveAs("path/to/your/processed_image.jpg");

        // Display the processed image
        imagePlus.show();
    }
}