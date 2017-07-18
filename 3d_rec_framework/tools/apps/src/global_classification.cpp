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
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <armadillo>
#include <pcl/common/time.h>

#include "opencv2\opencv.hpp"

 // regress a line through the given points and return error
std::tuple<double, arma::vec> regress(std::vector<std::tuple<int, int>>& pos)
{
	// fill the data in
	arma::mat x_val(1, pos.size());
	arma::vec w_val(pos.size());
	for (int i = 0; i < pos.size(); i++)
	{
		x_val(0, i) = std::get<0>(pos[i]);
		w_val(i) = std::get<1>(pos[i]);
	}

	// regress
	mlpack::regression::LinearRegression lr(x_val, w_val);

	// now calculate error
	arma::vec result(pos.size());
	lr.Predict(x_val, result);
	std::tuple<double, arma::vec> error_and_parameters(lr.ComputeError(x_val, w_val), lr.Parameters());
	return error_and_parameters;
}

// regress the variable t in the equation
// y = m * x + t
// when m is fixed
// for the given input values
double regress_t_with_fixed_m(std::vector<std::tuple<int, int>>& pos, double m)
{
	double n = pos.size();

	double accum = 0.0;
	for (int i = 0; i < n; i++)
	{
		accum += std::get<1>(pos[i]) - m * std::get<0>(pos[i]);
	}
	double error = 0.0;
	double t = accum / n;
	for (int j = 0; j < n; j++) {
		double tmp = (std::get<1>(pos[j]) - t) * (std::get<1>(pos[j]) - t);
		error += tmp;
	}

	return error / n;
}

double regress_split_at(std::vector<std::tuple<int, int>>& part_a, std::vector<std::tuple<int, int>>& part_b)
{
	double error_a = std::get<0>(regress(part_a));
	double error_b = regress_t_with_fixed_m(part_b, 0.0);
	return error_a + error_b;
}

int regression(boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>>& needle_width) {
	std::vector<std::tuple<int, int>> widths;
	for (int i = 0; i < needle_width->size(); i++) {
		widths.push_back(std::tuple<int, int>(std::get<0>(needle_width->at(i)), std::get<1>(needle_width->at(i))));
	}
	std::vector<double> errors;
	for (int j = 2; j < widths.size(); j++) {
		errors.push_back(regress_split_at(std::vector<std::tuple<int, int>>(widths.begin(), widths.begin() + j), std::vector<std::tuple<int, int>>(widths.begin() + j, widths.end())));
	}
	int error_min_index = 0;
	for (int k = 0; k < errors.size(); k++) {
		if (errors[k] < errors[error_min_index]) {
			error_min_index = k;
		}
	}
	int index = error_min_index;
	return index;
}

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
void processOCTFrame(cv::Mat imageGray, int number, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>>& needle_width) {
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
			//save bounding box width for finding the point where needle gets smaller
			needle_width->push_back(std::tuple<int, int, cv::Mat, cv::Mat>(number, labelStats.at<int>(i, cv::CC_STAT_WIDTH), filteredImage, labelInfo));
		}
	}
}

//-----------------------------------
//setup oct point cloud for alignment
//-----------------------------------
template<template<class > class DistT, typename PointT, typename FeatureT>
void
recognizeOCT(typename pcl::rec_3d_framework::GlobalNNCRHRecognizer<DistT, PointT, FeatureT> & global, std::string oct_dir, bool only_tip) {

	std::string oct_directory = getDirectoryPath(oct_dir);
	//count oct images
	int fileCount = countNumberOfFilesInDirectory(oct_directory, "%s*.bmp");
	int minFrameNumber = 0;
	int maxFrameNumber = fileCount;

	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
	boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>> needle_width(new std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>);
	cv::Mat imageGray;
	{
		pcl::ScopeTime t("Process OCT images");
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
			imageGray = cv::imread(oct_directory.c_str() + filename.str(), CV_LOAD_IMAGE_GRAYSCALE);

			processOCTFrame(imageGray, number, point_cloud_ptr, needle_width);

			cv::waitKey(10);
		}
		int end_index = needle_width->size();
		//regression to find cutting point where tip ends
		if (only_tip) {
			end_index = regression(needle_width);
		}
		//go through all frames
		for (int w = 0; w < end_index; w++) {
			std::tuple<int, int, cv::Mat, cv::Mat> tup = needle_width->at(w);
			MatToPoinXYZ(std::get<2>(tup), std::get<3>(tup), w, point_cloud_ptr, imageGray.rows, imageGray.cols);
		}
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
	boost::shared_ptr<std::vector<std::tuple<int, int, Eigen::Matrix4f, typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::ConstPtr, typename pcl::PointCloud<PointT>::Ptr>>> results;
	results = global.get_Id_Inliers_Transform_Output_Input_Crha();

	//print the resulting transform of the best match
	Eigen::Matrix4f result_transform(std::get<2>(results->at(0)));
	Eigen::Matrix3f result_rotation_matrix(result_transform.block(0, 0, 3, 3));
	Eigen::Vector3f result_euler_angles = result_rotation_matrix.eulerAngles(0, 1, 2);
	result_euler_angles *= 180 / M_PI;
	std::cout << "result euler angles: " << result_euler_angles << std::endl;
	std::cout << "rotation around z axis: " << std::abs(result_euler_angles.z()) - 90 << std::endl;

	//-----------------------------------
	//show the computed point clouds
	//-----------------------------------
	for (int i = 0; i < results->size(); i++) {
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler(std::get<3>(results->at(i)), 0, 0, 255);
		viewer->addPointCloud<pcl::PointXYZ>(std::get<3>(results->at(i)), rgb_handler, "sample cloud");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler3(point_cloud_ptr, 192, 192, 192);
		viewer->addPointCloud<pcl::PointXYZ>(point_cloud_ptr, rgb_handler3, "sample cloud 3");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler2(std::get<4>(results->at(i)), 255, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(std::get<4>(results->at(i)), rgb_handler2, "sample cloud 2");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler4(std::get<5>(results->at(i)), 0, 255, 0);
		viewer->addPointCloud<pcl::PointXYZ>(std::get<5>(results->at(i)), rgb_handler4, "sample cloud 4");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "sample cloud");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud 2");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud 3");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud 4");
		viewer->addCoordinateSystem(2.0);
		viewer->initCameraParameters();
		viewer->spin();
	}
}

//bin/pcl_global_classification -models_dir /directory/of/cad/model/in/ply/format -training_dir /directory/where/trained/models/should/be/saved -nn 10 -oct_dir /directory/to/oct/frames -only_tip 1

int
main(int argc, char ** argv)
{
	//------------------------------
	//parse command line arguments
	//------------------------------
	std::string path = "models/";
	std::string desc_name = "cvfh";
	std::string training_dir = "trained_models/";
	std::string oct_dir = "oct/";
	bool only_tip = false;
	int NN = 1;

	pcl::console::parse_argument(argc, argv, "-models_dir", path);
	pcl::console::parse_argument(argc, argv, "-training_dir", training_dir);
	//pcl::console::parse_argument(argc, argv, "-descriptor_name", desc_name);
	pcl::console::parse_argument(argc, argv, "-nn", NN);
	pcl::console::parse_argument(argc, argv, "-oct_dir", oct_dir);
	pcl::console::parse_argument(argc, argv, "-only_tip", only_tip);

	//pcl::console::parse_argument (argc, argv, "-z_dist", chop_at_z_);
	//pcl::console::parse_argument (argc, argv, "-tesselation_level", views_level_);

	//-----------------------------
	// generate views of CAD model
	//-----------------------------
	boost::shared_ptr<pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > mesh_source(new pcl::rec_3d_framework::MeshSource<pcl::PointXYZ>);
	mesh_source->setPath(path);
	mesh_source->setResolution(150);
	mesh_source->setTesselationLevel(1);
	mesh_source->setViewAngle(90.f);
	mesh_source->setRadiusSphere(.5f);
	mesh_source->setModelScale(1.f);
	mesh_source->generate(training_dir);

	boost::shared_ptr<pcl::rec_3d_framework::Source<pcl::PointXYZ> > cast_source;
	cast_source = boost::static_pointer_cast<pcl::rec_3d_framework::MeshSource<pcl::PointXYZ>> (mesh_source);

	//-----------------------
	//initialize normal estimator pointer
	//-----------------------
	boost::shared_ptr<pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal> > normal_estimator;
	normal_estimator.reset(new pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal>);
	normal_estimator->setCMR(true);
	normal_estimator->setDoVoxelGrid(true);
	normal_estimator->setRemoveOutliers(false);
	normal_estimator->setFactorsForCMR(3, 7);
	//leaf size 3cm, normals: use all neighbours in radius 30cm
	normal_estimator->setValuesForCMRFalse(0.03, 0.3);

	if (desc_name.compare("cvfh") == 0)
	{
		//----------------------------------------------------
		//initialize camera roll histogram estimator pointer
		//----------------------------------------------------
		boost::shared_ptr<pcl::rec_3d_framework::CRHEstimation<pcl::PointXYZ, pcl::VFHSignature308> > crh_estimator;
		crh_estimator.reset(new pcl::rec_3d_framework::CRHEstimation<pcl::PointXYZ, pcl::VFHSignature308>);
		crh_estimator->setNormalEstimator(normal_estimator);

		//-----------------------------------------------------------
		//initialize clustered viewpoint feature histogram pointer
		//-----------------------------------------------------------
		boost::shared_ptr<pcl::rec_3d_framework::CVFHEstimation<pcl::PointXYZ, pcl::VFHSignature308> > cvfh_estimator;
		cvfh_estimator.reset(new pcl::rec_3d_framework::CVFHEstimation<pcl::PointXYZ, pcl::VFHSignature308>);
		cvfh_estimator->setNormalEstimator(normal_estimator);
		//cast cvfh estimator to global estimator so that it can be used with crh
		boost::shared_ptr<pcl::rec_3d_framework::GlobalEstimator<pcl::PointXYZ, pcl::VFHSignature308> > cast_estimator;
		cast_estimator = boost::dynamic_pointer_cast<pcl::rec_3d_framework::CVFHEstimation<pcl::PointXYZ, pcl::VFHSignature308>> (cvfh_estimator);

		crh_estimator->setFeatureEstimator(cast_estimator);

		//--------------------------------
		//initialize global crh recognizer
		//--------------------------------
		pcl::rec_3d_framework::GlobalNNCRHRecognizer<Metrics::HistIntersectionUnionDistance, pcl::PointXYZ, pcl::VFHSignature308> global;
		global.setDataSource(cast_source);
		global.setTrainingDir(training_dir);
		global.setDescriptorName(desc_name);
		global.setFeatureEstimator(crh_estimator);
		global.setNN(NN);
		global.setICPIterations(100);
		global.setDOCRH(true);
		//computes descriptors / loads them
		global.initialize(false);

		//-------------------------------------
		//process oct images, alignment, etc.
		//-------------------------------------
		recognizeOCT<Metrics::HistIntersectionUnionDistance, pcl::PointXYZ, pcl::VFHSignature308>(global, oct_dir, only_tip);
	}
}
