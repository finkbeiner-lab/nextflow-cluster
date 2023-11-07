package com.finkbeiner;

import java.io.File;
import ij.io.Opener;
import ij.plugin.frame.RoiManager;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.gui.GenericDialog;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

import java.awt.Color;
import java.awt.TextField;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import mpicbg.ij.InverseTransformMapping;
import mpicbg.ij.Mapping;
import mpicbg.ij.SIFT;
import mpicbg.imagefeatures.Feature;
import mpicbg.imagefeatures.Filter;
import mpicbg.imagefeatures.FloatArray2D;
import mpicbg.imagefeatures.FloatArray2DSIFT;
import mpicbg.imagefeatures.ImageArrayConverter;
import mpicbg.models.AbstractAffineModel2D;
import mpicbg.models.AffineModel2D;
import mpicbg.models.PointMatch;
import mpicbg.models.RigidModel2D;
import mpicbg.models.SimilarityModel2D;
import mpicbg.models.TranslationModel2D;


public class Sift_Registration {
    

    static private class Param
	{
		final public FloatArray2DSIFT.Param sift = new FloatArray2DSIFT.Param();

		/**
		 * Closest/next closest neighbour distance ratio
		 */
		public float rod = 0.92f;

		/**
		 * Maximal allowed alignment error in px
		 */
		public float maxEpsilon = 25.0f;

		/**
		 * Inlier/candidates ratio
		 */
		public float minInlierRatio = 0.05f;

		/**
		 * Implemeted transformation models for choice
		 */
		final static public String[] modelStrings = new String[]{ "Translation", "Rigid", "Similarity", "Affine" };
		public int modelIndex = 1;

		public boolean interpolate = true;

		public boolean showInfo = false;
		
		public boolean showMatrix = false;
	}
	final static Param p = new Param();
	final static private ImageProcessor downScale( final ImageProcessor ip, final double s )
	{
		final FloatArray2D g = new FloatArray2D( ip.getWidth(), ip.getHeight() );
		ImageArrayConverter.imageProcessorToFloatArray2D( ip, g );

		final float sigma = ( float )Math.sqrt( 0.25 * 0.25 / s / s - 0.25 );
		final float[] kernel = Filter.createGaussianKernel( sigma, true );

		final FloatArray2D h = Filter.convolveSeparable( g, kernel, kernel );

		final FloatProcessor fp = new FloatProcessor( ip.getWidth(), ip.getHeight() );

		ImageArrayConverter.floatArray2DToFloatProcessor( h, fp );
		return ip.resize( ( int )( s * ip.getWidth() ) );
	}
    public static void main(String[] args) {
		final List< Feature > fs1 = new ArrayList< Feature >();
    	final List< Feature > fs2 = new ArrayList< Feature >();
        fs1.clear();
        fs2.clear();

        String parent_dir = "/gladstone/finkbeiner/robodata/Robo4Images/20230928-MsNeu-RGEDItau1/A8/";
		// get analysis dir

		// get images to align

		// Save images in AlignedImages

		// update coordinates, multiply by affine, result in version 1
        String src_file = parent_dir + "PID20230928_20230928-MsNeu-RGEDItau1_T0_0.0-0_A8_1_Confocal-GFP16_0_0_1.tif";
        String dst_file = parent_dir + "PID20230929_20230928-MsNeu-RGEDItau1_T1_12.0-0_A8_1_Confocal-GFP16_0_0_1.tif";
        String outputStackPath = "registered.tif";
        ImagePlus src_tile = IJ.openImage(src_file);
        ImageStack stack = new ImageStack(src_tile.getWidth(), src_tile.getHeight());

        // src_tile.show();

        
        // Add each image to the stack
        stack.addSlice(src_tile.getProcessor());
        // Optionally, set the title for each slice (image)
        stack.setSliceLabel(src_tile.getShortTitle(), stack.getSize());
        src_tile.close();

        ImagePlus dst_tile = IJ.openImage(dst_file);
        // Add each image to the stack
        // dst_tile.show();

        stack.addSlice(dst_tile.getProcessor());
        // Optionally, set the title for each slice (image)
        stack.setSliceLabel(dst_tile.getShortTitle(), stack.getSize());
        dst_tile.close();

        

        
        // Create an ImagePlus with the stacked images
        // ImagePlus imp = new ImagePlus("Stacked Images", stack);
        // Replace with the path to your input stack of images

        // ImageStack stack = imp.getStack();
        
        // Create a new ImagePlus for the aligned stack
        ImagePlus imp = new ImagePlus("Aligned Stack", stack);
        // imp.show();

        // ImageProcessor ip = alignedStack.getProcessor();
        // Initialize SIFT parameters
        p.sift.fdBins = 4;
        p.sift.fdSize = 8;
        p.sift.minOctaveSize = 64;
        p.sift.maxOctaveSize = 1024;
        p.sift.steps = 3;
        p.showInfo = false;
        p.modelIndex = 1;
        p.showMatrix = true;
        FloatArray2DSIFT t = new FloatArray2DSIFT(p.sift);
        
        // SIFT sft = new SIFT(t);
        // sft.extractFeatures(ip);


		// final ImageStack stack = imp.getStack();
		final ImageStack stackAligned = new ImageStack( stack.getWidth(), stack.getHeight() );

		final float vis_scale = 256.0f / imp.getWidth();
		ImageStack stackInfo = null;
		ImagePlus impInfo = null;

		if ( p.showInfo )
			stackInfo = new ImageStack(
					Math.round( vis_scale * stack.getWidth() ),
					Math.round( vis_scale * stack.getHeight() ) );

		final ImageProcessor firstSlice = stack.getProcessor( 1 );
		stackAligned.addSlice( null, firstSlice.duplicate() );
		stackAligned.getProcessor( 1 ).setMinAndMax( firstSlice.getMin(), firstSlice.getMax() );
		final ImagePlus impAligned = new ImagePlus( "Aligned 1 of " + stack.getSize(), stackAligned );
		impAligned.show();

		ImageProcessor ip1;
		ImageProcessor ip2 = stack.getProcessor( 1 );
		ImageProcessor ip3 = null;
		ImageProcessor ip4 = null;

		final FloatArray2DSIFT sift = new FloatArray2DSIFT( p.sift );
		final SIFT ijSIFT = new SIFT( sift );

		long start_time = System.currentTimeMillis();
		IJ.log( "Processing SIFT ..." );
		ijSIFT.extractFeatures( ip2, fs2 );
		IJ.log( " took " + ( System.currentTimeMillis() - start_time ) + "ms." );
		IJ.log( fs2.size() + " features extracted." );

		// downscale ip2 for visualisation purposes
		if ( p.showInfo )
			ip2 = downScale( ip2, vis_scale );

		AbstractAffineModel2D model;
		switch ( p.modelIndex )
		{
		case 0:
			model = new TranslationModel2D();
			break;
		case 1:
			model = new RigidModel2D();
			break;
		case 2:
			model = new SimilarityModel2D();
			break;
		case 3:
			model = new AffineModel2D();
			break;
		default:
			return;
		}
		final Mapping mapping = new InverseTransformMapping< AbstractAffineModel2D< ? > >( model );

		for ( int i = 1; i < stack.getSize(); ++i )
		{
			ip1 = ip2;
			ip2 = stack.getProcessor( i + 1 );

			fs1.clear();
			fs1.addAll( fs2 );
			fs2.clear();

			start_time = System.currentTimeMillis();
			IJ.log( "Processing SIFT ..." );
			ijSIFT.extractFeatures( ip2, fs2 );
			IJ.log( " took " + ( System.currentTimeMillis() - start_time ) + "ms." );
			IJ.log( fs2.size() + " features extracted." );

			start_time = System.currentTimeMillis();
			System.out.print( "identifying correspondences using brute force ..." );
			final Vector< PointMatch > candidates =
				FloatArray2DSIFT.createMatches( fs2, fs1, 1.5f, null, Float.MAX_VALUE, p.rod );
			System.out.println( " took " + ( System.currentTimeMillis() - start_time ) + "ms" );

			IJ.log( candidates.size() + " potentially corresponding features identified" );

			/**
			 * draw all correspondence candidates
			 */
			if (p.showInfo )
			{
				ip2 = downScale( ip2, vis_scale );

				ip3 = ip1.convertToRGB().duplicate();
				ip4 = ip2.convertToRGB().duplicate();
				ip3.setColor( Color.red );
				ip4.setColor( Color.red );

				ip3.setLineWidth( 2 );
				ip4.setLineWidth( 2 );
				for ( final PointMatch m : candidates )
				{
					final double[] m_p1 = m.getP1().getL();
					final double[] m_p2 = m.getP2().getL();

					ip3.drawDot( ( int )Math.round( vis_scale * m_p2[ 0 ] ), ( int )Math.round( vis_scale * m_p2[ 1 ] ) );
					ip4.drawDot( ( int )Math.round( vis_scale * m_p1[ 0 ] ), ( int )Math.round( vis_scale * m_p1[ 1 ] ) );
				}
			}

			final Vector< PointMatch > inliers = new Vector< PointMatch >();

			// TODO Implement other models for choice
			AbstractAffineModel2D< ? > currentModel;
			switch ( p.modelIndex )
			{
			case 0:
				currentModel = new TranslationModel2D();
				break;
			case 1:
				currentModel = new RigidModel2D();
				break;
			case 2:
				currentModel = new SimilarityModel2D();
				break;
			case 3:
				currentModel = new AffineModel2D();
				break;
			default:
				return;
			}

			boolean modelFound;
			try
			{
				modelFound = currentModel.filterRansac(
						candidates,
						inliers,
						1000,
						p.maxEpsilon,
						p.minInlierRatio );
			}
			catch ( final Exception e )
			{
				modelFound = false;
				System.err.println( e.getMessage() );
			}
			if ( modelFound )
			{
				if ( p.showInfo )
				{
					ip3.setColor( Color.green );
					ip4.setColor( Color.green );
					ip3.setLineWidth( 2 );
					ip4.setLineWidth( 2 );
					for ( final PointMatch m : inliers )
					{
						final double[] m_p1 = m.getP1().getL();
						final double[] m_p2 = m.getP2().getL();

						ip3.drawDot( ( int )Math.round( vis_scale * m_p2[ 0 ] ), ( int )Math.round( vis_scale * m_p2[ 1 ] ) );
						ip4.drawDot( ( int )Math.round( vis_scale * m_p1[ 0 ] ), ( int )Math.round( vis_scale * m_p1[ 1 ] ) );
					}
				}

				/**
				 *       [ x']   [  m00  m01  m02  ] [ x ]   [ m00x + m01y + m02 ]
						[ y'] = [  m10  m11  m12  ] [ y ] = [ m10x + m11y + m12 ]
						[ 1 ]   [   0    0    1   ] [ 1 ]   [         1         ]
				 *
				 */
				model.concatenate( currentModel );

				if ( p.showMatrix )
				{
					IJ.log("Transformation Matrix: " + currentModel.createAffine() );
				}
			
			}

//			ImageProcessor alignedSlice = stack.getProcessor( i + 1 ).duplicate();
			final ImageProcessor originalSlice = stack.getProcessor( i + 1 );
			originalSlice.setInterpolationMethod( ImageProcessor.BILINEAR );
			final ImageProcessor alignedSlice = originalSlice.createProcessor( stack.getWidth(), stack.getHeight() );
			alignedSlice.setMinAndMax( originalSlice.getMin(), originalSlice.getMax() );

			if ( p.interpolate )
				mapping.mapInterpolated( originalSlice, alignedSlice );
			else
				mapping.map( originalSlice, alignedSlice );

			IJ.log("mapping transform" + mapping.getTransform().toString());

			stackAligned.addSlice( null, alignedSlice );
			if ( p.showInfo )
			{
				ImageProcessor tmp;
				tmp = ip3.createProcessor( stackInfo.getWidth(), stackInfo.getHeight() );
				tmp.insert( ip3, 0, 0 );
				stackInfo.addSlice( null, tmp ); // fixing silly 1 pixel size missmatches
				tmp = ip4.createProcessor( stackInfo.getWidth(), stackInfo.getHeight() );
				tmp.insert( ip4, 0, 0 );
				stackInfo.addSlice( null, tmp );
				if ( i == 1 )
				{
					impInfo = new ImagePlus( "Alignment info", stackInfo );
					impInfo.show();
				}
				impInfo.setStack( "Alignment info", stackInfo );
				final int currentSlice = impInfo.getSlice();
				impInfo.setSlice( stackInfo.getSize() );
				impInfo.setSlice( currentSlice );
				impInfo.updateAndDraw();
			}
			impAligned.setStack( "Aligned " + stackAligned.getSize() + " of " + stack.getSize(), stackAligned );
			final int currentSlice = impAligned.getSlice();
			impAligned.setSlice( stack.getSize() );
			impAligned.setSlice( currentSlice );
			impAligned.updateAndDraw();
		}
		IJ.log( "Done." );
    }
}

