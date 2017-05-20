/*
 * test_training.cpp
 *
 *  Created on: Mar 9, 2012
 *      Author: aitor
 */

#include <pcl/pcl_macros.h>
#include <pcl/apps/3d_rec_framework/pipeline/global_nn_classifier.h>
#include <pcl/apps/3d_rec_framework/pipeline/global_nn_recognizer_crh.h>
#include <pcl/apps/3d_rec_framework/pipeline/global_nn_recognizer_cvfh.h>
#include <pcl/apps/3d_rec_framework/pc_source/mesh_source.h>
#include <pcl/apps/3d_rec_framework/feature_wrapper/global/vfh_estimator.h>
#include <pcl/apps/3d_rec_framework/feature_wrapper/global/esf_estimator.h>
#include <pcl/apps/3d_rec_framework/feature_wrapper/global/cvfh_estimator.h>
#include <pcl/apps/3d_rec_framework/feature_wrapper/global/crh_estimator.h>
#include <pcl/apps/3d_rec_framework/tools/openni_frame_source.h>
#include <pcl/apps/3d_rec_framework/utils/metrics.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/apps/dominant_plane_segmentation.h>
#include <pcl/console/parse.h>
#include <tuple>
#include <pcl/apps/3d_rec_framework/pc_source/source.h>
#include "pcl/apps/3d_rec_framework/utils/util.h"

#include "opencv2\opencv.hpp"



//------------------------------------------------------------
//convert labelled image (opencv matrix) to points for cloud
//------------------------------------------------------------
void MatToPoinXYZ(cv::Mat& OpencVPointCloud, cv::Mat& labelInfo, int z, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, int height, int width)
{
	int volumeBScans = 128;
	//get the infos for the bounding box
	int x = labelInfo.at<int>(0, cv::CC_STAT_LEFT);
	int y = labelInfo.at<int>(0, cv::CC_STAT_TOP);
	int labelWidth = labelInfo.at<int>(0, cv::CC_STAT_WIDTH);
	int labelHeight = labelInfo.at<int>(0, cv::CC_STAT_HEIGHT);
	//go through points in bounding box 
	for (int i = x; i < x + labelWidth; i++) {
		//indicate if first point with intensity = 1 in row has been found
		bool firstNotFound = true;
		//position of last point with intensity = 1 in row
		int lastPointPosition = 0;
		for (int j = y; j < y + labelHeight; j++)
		{
			if (OpencVPointCloud.at<unsigned char>(j, i) >= 1.0f) {
				if (firstNotFound) {
					firstNotFound = false;
				}
				lastPointPosition = j;
			}
		}
		if (!firstNotFound) {
			//add the last point with intensity = 1 in row to the point cloud
			pcl::PointXYZ point;
			point.x = (float)i / width * 3.0f;
			point.y = (float)lastPointPosition / height * 2.0f;
			point.z = (float)z / volumeBScans * 2.6f;
			point_cloud_ptr->points.push_back(point);
		}
	}
}

//----------------------------------------------
//process the OCT frame to get a labelled image
//----------------------------------------------
void processOCTFrame(cv::Mat imageGray, int number, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr) {
	//flip and transpose the image
	cv::Mat transposedOCTimage;
	cv::flip(imageGray, imageGray, 0);

	//set a threshold (0.26)
	cv::Mat thresholdedImage;
	cv::threshold(imageGray, thresholdedImage, 0.26 * 255, 1, 0);

	//use a median blur filter
	cv::Mat filteredImage;
	cv::medianBlur(thresholdedImage, filteredImage, 3);

	//label the image
	cv::Mat labelledImage;
	cv::Mat labelStats;
	cv::Mat labelCentroids;
	int numLabels = cv::connectedComponentsWithStats(filteredImage, labelledImage, labelStats, labelCentroids);

	//for every label with more than 400 points process it further for adding points to the cloud
	for (int i = 1; i < numLabels; i++) {
		if (labelStats.at<int>(i, cv::CC_STAT_AREA) > 400) {
			cv::Mat labelInfo = labelStats.row(i);
			MatToPoinXYZ(filteredImage, labelInfo, number, point_cloud_ptr, thresholdedImage.rows, thresholdedImage.cols);
		}
	}
}

//-----------------------------------
//setup oct point cloud for alignment
//-----------------------------------
template<template<class > class DistT, typename PointT, typename FeatureT>
void
recognizeOCT(typename pcl::rec_3d_framework::GlobalNNCRHRecognizer<DistT, PointT, FeatureT> & global, std::string oct_dir) {
	
	std::string oct_directory = getDirectoryPath(oct_dir);
	//count oct images
	int fileCount = countNumberOfFilesInDirectory(oct_directory, "%s*.bmp");
	int minFrameNumber = 0;
	int maxFrameNumber = fileCount;

	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
	//	go through all frames
	for (int number = minFrameNumber; number < maxFrameNumber; number++)
	{
		//get the next frame
		std::stringstream filename;
		if (number < 100) {
			filename << "0";
		}
		if (number < 10) {
			filename << "0";
		}
		filename << number << ".bmp";
		//read the image in grayscale
		cv::Mat imageGray = cv::imread(oct_directory.c_str() + filename.str(), CV_LOAD_IMAGE_GRAYSCALE);

		processOCTFrame(imageGray, number, point_cloud_ptr);

		cv::waitKey(10);

	}
	//set the oct cloud as input for the global crh pipeline
	global.setInputCloud(point_cloud_ptr);
	//alignment happens here
	global.recognize();

	//get the transformation matrices
	boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms;
	transforms = global.getTransforms();
	//get the alignment results ordered by number of inliers from ICP 
	//1) id of view, 2) number of inliers, 3) transformation matrix, 4) output (cad model with applied transformation matrix), 5) input (cad model)
	boost::shared_ptr<std::vector<std::tuple<int, int, Eigen::Matrix4f, typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::ConstPtr>>> results;
	results = global.get_Id_Inliers_Transform_Output_Input();

	//-----------------------------------
	//show the computed point clouds
	//-----------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler(std::get<3>(results->at(0)), 0, 0, 255);
	viewer->addPointCloud<pcl::PointXYZ>(std::get<3>(results->at(0)), rgb_handler, "sample cloud");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler3(point_cloud_ptr, 192, 192, 192);
	viewer->addPointCloud<pcl::PointXYZ>(point_cloud_ptr, rgb_handler3, "sample cloud 3");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler2(std::get<4>(results->at(0)), 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(std::get<4>(results->at(0)), rgb_handler2, "sample cloud 2");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud 2");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud 3");
	viewer->addCoordinateSystem(2.0);
	viewer->initCameraParameters();
	viewer->spin();
}

//bin/pcl_global_classification -models_dir /directory/of/cad/model/in/ply/format -descriptor_name cvfh -training_dir /directory/where/trained/models/should/be/saved -nn 10 -oct_dir /directory/to/oct/frames

int
main (int argc, char ** argv)
{
	//------------------------------
	//parse command line arguments
	//------------------------------
  std::string path = "models/";
  std::string desc_name = "cvfh";
  std::string training_dir = "trained_models/";
  std::string oct_dir = "oct/";
  int NN = 1;

  pcl::console::parse_argument (argc, argv, "-models_dir", path);
  pcl::console::parse_argument (argc, argv, "-training_dir", training_dir);
  pcl::console::parse_argument (argc, argv, "-descriptor_name", desc_name);
  pcl::console::parse_argument (argc, argv, "-nn", NN);
  pcl::console::parse_argument (argc, argv, "-oct_dir", oct_dir);

  //pcl::console::parse_argument (argc, argv, "-z_dist", chop_at_z_);
  //pcl::console::parse_argument (argc, argv, "-tesselation_level", views_level_);

  //-----------------------------
  // generate views of CAD model
  //-----------------------------
  boost::shared_ptr<pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > mesh_source (new pcl::rec_3d_framework::MeshSource<pcl::PointXYZ>);
  mesh_source->setPath (path);
  mesh_source->setResolution (150);
  mesh_source->setTesselationLevel (1);
  mesh_source->setViewAngle (57.f);
  mesh_source->setRadiusSphere (1.5f);
  mesh_source->setModelScale (1.f);
  mesh_source->generate (training_dir);

  boost::shared_ptr<pcl::rec_3d_framework::Source<pcl::PointXYZ> > cast_source;
  cast_source = boost::static_pointer_cast<pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > (mesh_source);

  //-----------------------
  //initialize normal estimator pointer
  //-----------------------
  boost::shared_ptr<pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal> > normal_estimator;
  normal_estimator.reset (new pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal>);
  normal_estimator->setCMR (true);
  normal_estimator->setDoVoxelGrid (true);
  normal_estimator->setRemoveOutliers (true);
  normal_estimator->setFactorsForCMR (3, 7);

  if (desc_name.compare ("cvfh") == 0)
  {
	  //----------------------------------------------------
	  //initialize camera roll histogram estimator pointer
	  //----------------------------------------------------
    boost::shared_ptr<pcl::rec_3d_framework::CRHEstimation<pcl::PointXYZ, pcl::VFHSignature308> > crh_estimator;
    crh_estimator.reset (new pcl::rec_3d_framework::CRHEstimation<pcl::PointXYZ, pcl::VFHSignature308>);
    crh_estimator->setNormalEstimator (normal_estimator);

	//-----------------------------------------------------------
	//initialize clustered viewpoint feature histogram pointer
	//-----------------------------------------------------------
	boost::shared_ptr<pcl::rec_3d_framework::CVFHEstimation<pcl::PointXYZ, pcl::VFHSignature308> > cvfh_estimator;
	cvfh_estimator.reset(new pcl::rec_3d_framework::CVFHEstimation<pcl::PointXYZ, pcl::VFHSignature308>);
	cvfh_estimator->setNormalEstimator(normal_estimator);
	//cast cvfh estimator to global estimator so that it can be used with crh
    boost::shared_ptr<pcl::rec_3d_framework::GlobalEstimator<pcl::PointXYZ, pcl::VFHSignature308> > cast_estimator;
    cast_estimator = boost::dynamic_pointer_cast<pcl::rec_3d_framework::CVFHEstimation<pcl::PointXYZ, pcl::VFHSignature308> > (cvfh_estimator);

	crh_estimator->setFeatureEstimator(cast_estimator);
	
	//--------------------------------
	//initialize global crh recognizer
	//--------------------------------
    pcl::rec_3d_framework::GlobalNNCRHRecognizer<Metrics::HistIntersectionUnionDistance, pcl::PointXYZ, pcl::VFHSignature308> global;
    global.setDataSource (cast_source);
    global.setTrainingDir (training_dir);
    global.setDescriptorName (desc_name);
    global.setFeatureEstimator (crh_estimator);
    global.setNN (NN);
	global.setICPIterations(1);
	//computes descriptors / loads them
    global.initialize (false);
	
	//-------------------------------------
	//process oct images, alignment, etc.
	//-------------------------------------
	recognizeOCT<Metrics::HistIntersectionUnionDistance, pcl::PointXYZ, pcl::VFHSignature308>(global, oct_dir);
  }
}
