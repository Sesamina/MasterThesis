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
#include <pcl/apps/3d_rec_framework/utils/vtk_model_sampling.h>

#include <armadillo>

#include "pcl/apps/3d_rec_framework/utils/regression.h"

#include <pcl/common/time.h>

#include "opencv2\opencv.hpp"

 //fixed number of OCT images
int numFrames = 128;

//------------------------------------------------------------
//convert labelled image (opencv matrix) to points for cloud
//------------------------------------------------------------
void MatToPointXYZ(cv::Mat& OpencVPointCloud, cv::Mat& labelInfo, int z, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr& peak_points, int height, int width)
{
	//get the infos for the bounding box
	int x = labelInfo.at<int>(0, cv::CC_STAT_LEFT);
	int y = labelInfo.at<int>(0, cv::CC_STAT_TOP);
	int labelWidth = labelInfo.at<int>(0, cv::CC_STAT_WIDTH);
	int labelHeight = labelInfo.at<int>(0, cv::CC_STAT_HEIGHT);
	//current peak point
	pcl::PointXYZ peak_point(0, 0, 0);
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
			point.z = (float)z / numFrames * 2.6f;
			point_cloud_ptr->points.push_back(point);
			//smallest y value is the peak point
			if (point.y > peak_point.y) {
				peak_point = point;
			}
		}
	}
	peak_points->points.push_back(peak_point);
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

//----------------------------------------
//compute needle direction
//----------------------------------------
std::pair<Eigen::Vector3f, Eigen::Vector3f> computeNeedleDirection(pcl::PointCloud<pcl::PointXYZ>::Ptr& peak_points) {
	std::vector<Eigen::Vector3f> peak_positions;
	for (int i = 0; i < peak_points->points.size(); i++) {
		pcl::PointXYZ point = peak_points->points.at(i);
		Eigen::Vector3f eigenPoint(point.x, point.y, point.z);
		peak_positions.push_back(eigenPoint);
	}
	return fitLine(peak_positions);
}

//--------------------------------------
//compute needle rotation
//--------------------------------------
Eigen::Matrix3f computeNeedleRotation(std::pair<Eigen::Vector3f, Eigen::Vector3f> direction) {
	Eigen::Vector3f zRotation = std::get<1>(direction);
	Eigen::Vector3f up(0.0f, 1.0f, 0.0f);
	Eigen::Vector3f xRotation = up.cross(zRotation);
	xRotation.normalize();
	Eigen::Vector3f yRotation = zRotation.cross(xRotation);
	yRotation.normalize();
	Eigen::Matrix3f rotation;
	rotation << xRotation.x(), yRotation.x(), zRotation.x(),
		xRotation.y(), yRotation.y(), zRotation.y(),
		xRotation.z(), yRotation.z(), zRotation.z();
	return rotation;
}

//------------------------------------
//compute needle translation
//------------------------------------
Eigen::Vector3f computeNeedleTranslation(float tangencyPoint, Eigen::Vector3f pointOnOCTCloud, Eigen::Vector3f direction, float halfModelSize) {
	//needle model 1.5?
	if (direction.z() < 0) {
		direction *= -1;
	}
	Eigen::Vector3f translation = pointOnOCTCloud;
	float dist = std::abs(pointOnOCTCloud.z() - tangencyPoint);
	float mult = std::abs(dist / pointOnOCTCloud.z());
	if (pointOnOCTCloud.z() < tangencyPoint) {
		while (translation.z() < tangencyPoint) {
			translation += direction * mult;
		}
	}
	else if (pointOnOCTCloud.z() > tangencyPoint) {
		while (translation.z() > tangencyPoint) {
			translation -= direction * mult;
		}
	}
	translation -= (translation.z() / halfModelSize) * direction;
	return translation;
}

//-----------------------------------
//setup oct point cloud for alignment
//-----------------------------------
boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>>
recognizeOCT(pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr& peak_points, std::string oct_dir, bool only_tip) {

	std::string oct_directory = getDirectoryPath(oct_dir);
	//count oct images
	int fileCount = countNumberOfFilesInDirectory(oct_directory, "%s*.bmp");
	int minFrameNumber = 0;
	int maxFrameNumber = fileCount;

	//tuple with frame number, bounding box width, filteredImage, labelInfo
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

		//---------------------------------------------
		//optionally cut needle tip off
		//---------------------------------------------
		int end_index = needle_width->size();
		//regression to find cutting point where tip ends
		if (only_tip) {
			end_index = regression(needle_width);
		}
		//go through all frames
		for (int w = 0; w < end_index; w++) {
			std::tuple<int, int, cv::Mat, cv::Mat> tup = needle_width->at(w);
			MatToPointXYZ(std::get<2>(tup), std::get<3>(tup), w, point_cloud_ptr, peak_points, imageGray.rows, imageGray.cols);
		}
	}
	return needle_width;
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
	//use only needle tip or not
	bool only_tip = false;
	//use shift algorithm or icp
	bool shift = false;
	//number of nearest neighbours
	int NN = 1;

	pcl::console::parse_argument(argc, argv, "-models_dir", path);
	pcl::console::parse_argument(argc, argv, "-training_dir", training_dir);
	//pcl::console::parse_argument(argc, argv, "-descriptor_name", desc_name);
	pcl::console::parse_argument(argc, argv, "-nn", NN);
	pcl::console::parse_argument(argc, argv, "-oct_dir", oct_dir);
	pcl::console::parse_argument(argc, argv, "-only_tip", only_tip);
	pcl::console::parse_argument(argc, argv, "-shift", shift);


	//-------------------------------------
	//process OCT images
	//-------------------------------------

	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr peak_points(new pcl::PointCloud<pcl::PointXYZ>);
	boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>> needle_width = recognizeOCT(point_cloud_ptr, peak_points, oct_dir, only_tip);

	//-------------------------------
	//shifting algorithm - TODO: apply rotation of few degrees in every direction
	//-------------------------------
	if (shift) {

		//-------------------------------
		//process the CAD mdoel
		//-------------------------------
		pcl::PointCloud<pcl::PointXYZ>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized(new pcl::PointCloud<pcl::PointXYZ>());
		generatePointCloudFromModel(modelCloud, model_voxelized, path);

		//compute the 3d direction of the needle
		std::pair<Eigen::Vector3f, Eigen::Vector3f> direction = computeNeedleDirection(peak_points);
		std::cout << "origin: " << std::endl << std::get<0>(direction) << std::endl << "direction: " << std::endl << std::get<1>(direction) << std::endl;

		//compute the 3d rotation of the needle
		Eigen::Matrix3f rotation = computeNeedleRotation(direction);
		std::cout << "rotation matrix: " << std::endl << rotation << std::endl;

		//compute 3d translation of the needle
		float tangencyPoint = regression(needle_width) / (float)numFrames * 2.6f; //scaling
		std::cout << "tangency point: " << tangencyPoint << std::endl;
		Eigen::Vector3f initialTranslation = computeNeedleTranslation(tangencyPoint, std::get<0>(direction), std::get<1>(direction), getModelSize(modelCloud) / 2);
		std::cout << "translation: " << std::endl << initialTranslation << std::endl;

		//--------------TEST--------------
		Eigen::Matrix3f dummyRotationZ;
		float angle = -30 * M_PI / 180.0f;
		dummyRotationZ << std::cos(angle), -std::sin(angle), 0, std::sin(angle), std::cos(angle), 0, 0, 0, 1;
		rotation *= dummyRotationZ;
		//--------------TEST--------------

		//build transformation matrix
		Eigen::Matrix4f transformation;
		transformation.block(0, 0, 3, 3) = rotation;
		transformation.col(3).head(3) = initialTranslation;
		transformation.row(3) << 0, 0, 0, 1;
		//subtract needle radius in y direction to get model to same hight as oct cloud
		float needleDiameter = 0.31f;
		transformation(1, 3) -= needleDiameter / 2;


		pcl::PointCloud<pcl::PointXYZ>::Ptr modelTransformed(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::transformPointCloud(*modelCloud, *modelTransformed, transformation);



		//--------------------------------
		//visualization
		//--------------------------------
		pcl::PointCloud<pcl::PointXYZ>::Ptr origin_ptr(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointXYZ point2;
		point2.x = std::get<0>(direction).x() + 2 * std::get<1>(direction).x();
		point2.y = std::get<0>(direction).y() + 2 * std::get<1>(direction).y();
		point2.z = std::get<0>(direction).z() + 2 * std::get<1>(direction).z();
		pcl::PointXYZ point3;
		point3.x = std::get<0>(direction).x() - 2 * std::get<1>(direction).x();
		point3.y = std::get<0>(direction).y() - 2 * std::get<1>(direction).y();
		point3.z = std::get<0>(direction).z() - 2 * std::get<1>(direction).z();
		origin_ptr->points.push_back(point2);
		origin_ptr->points.push_back(point3);
		//show model
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler(model_voxelized, 0, 0, 255);
		viewer->addPointCloud<pcl::PointXYZ>(model_voxelized, rgb_handler, "model");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler2(origin_ptr, 255, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(origin_ptr, rgb_handler2, "test points");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler3(point_cloud_ptr, 0, 255, 0);
		viewer->addPointCloud<pcl::PointXYZ>(point_cloud_ptr, rgb_handler3, "oct cloud");
		/*pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler4(peak_points, 0, 255, 255);
		viewer->addPointCloud<pcl::PointXYZ>(peak_points, rgb_handler4, "peak points");*/
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler5(modelTransformed, 0, 255, 255);
		viewer->addPointCloud<pcl::PointXYZ>(modelTransformed, rgb_handler5, "model transformed");
		viewer->addLine(point2, point3, "line");
		viewer->addCoordinateSystem(2.0);
		viewer->initCameraParameters();
		viewer->spin();
	}
	//----------------------------------
	//CAD model recognition / 6DOF pose estimation algorithm from Aldoma paper
	else {
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

			//------------------------------------------
			//final adjustments and start of cvfh/crh/ipc algorithm
			//------------------------------------------
			//set the oct cloud as input for the global crh pipeline
			global.setInputCloud(point_cloud_ptr);
			//alignment happens here
			global.recognize();

			//get the transformation matrices
			boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms;
			transforms = global.getTransforms();

			//get the alignment results ordered by number of inliers from ICP 
			//1) id of view, 2) number of inliers, 3) transformation matrix, 4) output (cad model with applied transformation matrix), 5) input (cad model)
			boost::shared_ptr<std::vector<std::tuple<int, int, Eigen::Matrix4f, typename pcl::PointCloud<pcl::PointXYZ>::Ptr, typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr, typename pcl::PointCloud<pcl::PointXYZ>::Ptr>>> results;
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
	}
}
