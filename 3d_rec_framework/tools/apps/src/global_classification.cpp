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
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection.h>
#include <pcl/registration/default_convergence_criteria.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/distances.h>

#include <armadillo>

#include "pcl/apps/3d_rec_framework/utils/regression.h"

#include <pcl/common/time.h>

#include "opencv2\opencv.hpp"

#include "pcl/apps/3d_rec_framework/utils/graphUtils/GraphUtils.h"

 //fixed number of OCT images
#define NUM_FRAMES 128
 //scale of OCT cube
#define SCALE_X 2.7
#define SCALE_Y 2.4
#define SCALE_Z 3.0

//-------------------------------------
//helper method to generate a PointXYZ
//-------------------------------------
void generatePoint(pcl::PointXYZ& point, float x, float y, float z, float width, float height) {
	point.x = (float)x / width * SCALE_X;
	point.y = (float)y / height * SCALE_Y;
	point.z = (float)z / NUM_FRAMES * SCALE_Z;
}

//------------------------------------------------------------
//convert labelled image (opencv matrix) to points for cloud
//------------------------------------------------------------
void MatToPointXYZ(cv::Mat& OpencVPointCloud, cv::Mat& labelInfo, std::vector<cv::Point>& elipsePoints, int z, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, int height, int width)
{
	//get the infos for the bounding box
	int x = labelInfo.at<int>(0, cv::CC_STAT_LEFT);
	int y = labelInfo.at<int>(0, cv::CC_STAT_TOP);
	int labelWidth = labelInfo.at<int>(0, cv::CC_STAT_WIDTH);
	int labelHeight = labelInfo.at<int>(0, cv::CC_STAT_HEIGHT);
	int leftHeight = 0;
	int rightHeight = 0;
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
				if (i == x) {
					leftHeight = j;
				}
				if (i == x + labelWidth - 1) {
					rightHeight = j;
				}
			}
		}
		if (!firstNotFound) {
			//add the last point with intensity = 1 in row to the point cloud
			pcl::PointXYZ point;
			generatePoint(point, i, lastPointPosition, z, width, height);
			point_cloud_ptr->points.push_back(point);
			elipsePoints.push_back(cv::Point(i, lastPointPosition));
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

	//----------------------------------------
	//compute needle direction
	//----------------------------------------
	std::pair<Eigen::Vector3f, Eigen::Vector3f> computeNeedleDirection(pcl::PointCloud<pcl::PointXYZ>::Ptr& peak_points) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr peak_inliers(new pcl::PointCloud<pcl::PointXYZ>);
		std::vector<int> inliers = getInliers(peak_points);
		for (int i = 0; i < inliers.size(); i++) {
			peak_inliers->push_back(peak_points->at(inliers.at(i)));
		}
		std::vector<Eigen::Vector3f> peak_positions;
		for (int i = 0; i < peak_inliers->points.size(); i++) {//for RANSAC use peak_inliers, else peak_points
			pcl::PointXYZ point = peak_inliers->points.at(i); //for RANSAC use peak_inliers, else peak_points
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
		if (direction.z() < 0) {
			direction *= -1;
		}
		Eigen::Vector3f translation = pointOnOCTCloud;
		float dist = std::abs(pointOnOCTCloud.z() - tangencyPoint);
		float mult = std::abs(dist / direction.z());
		if (pointOnOCTCloud.z() < tangencyPoint) {
			translation += direction * mult;
		}
		else if (pointOnOCTCloud.z() > tangencyPoint) {
			translation -= direction * mult;
		}
		translation -= (halfModelSize / direction.z()) * direction;
		return translation;
	}

	//------------------------------------------------
	//rotate point cloud around z axis by given angle
	//------------------------------------------------
	Eigen::Matrix3f rotateByAngle(float angleInDegrees, Eigen::Matrix3f currentRotation) {
		Eigen::Matrix3f rotationZ;
		Eigen::Matrix3f finalRotation = currentRotation;
		float angle = angleInDegrees * M_PI / 180.0f;
		rotationZ << std::cos(angle), -std::sin(angle), 0, std::sin(angle), std::cos(angle), 0, 0, 0, 1;
		finalRotation *= rotationZ;
		return finalRotation;
	}

	//---------------------------------------------------------
	// compute translation given how much it should be shifted
	//---------------------------------------------------------
	Eigen::Vector3f shiftByValue(float shift, Eigen::Vector3f currentTranslation, Eigen::Vector3f direction) {
		Eigen::Vector3f finalTranslation = currentTranslation;
		finalTranslation += direction * (shift / direction.z());
		return finalTranslation;
	}

	//-----------------------------------------------------------------
	// build transformation matrix from given rotation and translation
	//-----------------------------------------------------------------
	Eigen::Matrix4f buildTransformationMatrix(Eigen::Matrix3f rotation, Eigen::Vector3f translation) {
		Eigen::Matrix4f transformation;
		transformation.block(0, 0, 3, 3) = rotation;
		transformation.col(3).head(3) = translation;
		transformation.row(3) << 0, 0, 0, 1;
		return transformation;
	}

	//-----------------------------
	// compute correspondences
	//-----------------------------
	float computeCorrespondences(Eigen::Matrix4f& guess, pcl::PointCloud<pcl::PointXYZ>::Ptr input, pcl::PointCloud<pcl::PointXYZ>::Ptr target) {
		// Point cloud containing the correspondences of each point in <input, indices>
		pcl::PointCloud<pcl::PointXYZ>::Ptr input_transformed(new pcl::PointCloud<pcl::PointXYZ>);

		// If the guessed transformation is non identity
		if (guess != Eigen::Matrix4f::Identity())
		{
			input_transformed->resize(input->size());
			// Apply passed transformation
			pcl::transformPointCloud(*input, *input_transformed, guess);
		}
		else
			*input_transformed = *input;

		pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>::Ptr correspondence_estimation(new pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>);
		// Pass in the default target for the Correspondence Estimation code
		correspondence_estimation->setInputTarget(target);

		// Set the source
		correspondence_estimation->setInputSource(input_transformed);
		boost::shared_ptr<pcl::Correspondences> correspondences(new pcl::Correspondences);
		// Estimate correspondences ---------- maxDistance: VoxelSize * 2
		/*#1)*/ correspondence_estimation->determineCorrespondences(*correspondences, 0.03f * 2);
		// /*#2)*/ correspondence_estimation->determineReciprocalCorrespondences(*correspondences, 0.03f * 2.f);
		boost::shared_ptr<pcl::Correspondences> temp_correspondences(new pcl::Correspondences(*correspondences));
		/*#3)#4)*/ /*pcl::registration::CorrespondenceRejectorDistance::Ptr rejector_distance(new pcl::registration::CorrespondenceRejectorDistance);
		rejector_distance->setInputCorrespondences(temp_correspondences);
		rejector_distance->setMaximumDistance(0.03f);
		rejector_distance->getCorrespondences(*correspondences);*/
		/*#5)#6)*/ pcl::registration::CorrespondenceRejectorMedianDistance::Ptr rejector_median(new pcl::registration::CorrespondenceRejectorMedianDistance);
		rejector_median->setInputCorrespondences(temp_correspondences);
		rejector_median->getCorrespondences(*correspondences);
		/*#7)#8)*//*pcl::registration::CorrespondenceRejectorOneToOne::Ptr rejector_oneToOne(new pcl::registration::CorrespondenceRejectorOneToOne);
		rejector_oneToOne->setInputCorrespondences(temp_correspondences);
		rejector_oneToOne->getCorrespondences(*correspondences);*/

		/*#9)#10)*//*pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ>::Ptr rejector_sampleConsensus(new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ>);
		rejector_sampleConsensus->setInputCorrespondences(temp_correspondences);
		rejector_sampleConsensus->setInputSource(input);
		rejector_sampleConsensus->setInputTarget(target);
		rejector_sampleConsensus->getCorrespondences(*correspondences);*/

		//if (use_error) {
			//float manhattan_sum = 0.0f;
			////Manhattan distance L1
			//for (auto& n : *correspondences) {
			//	manhattan_sum += n.distance;
			//}
			//manhattan_sum /= correspondences->size();
			//return manhattan_sum;

			/*double l1_sum = 0.0f;
			double l2_sum = 0.0f;
			double l2Sqr_sum = 0.0f;
			for (auto& n : *correspondences) {
				Eigen::Vector4f query(input_transformed->points.at(n.index_query).x, input_transformed->points.at(n.index_query).y, input_transformed->points.at(n.index_query).z, 0.0f);
				Eigen::Vector4f match(target->points.at(n.index_match).x, target->points.at(n.index_match).y, target->points.at(n.index_match).z, 0.0f);
				l1_sum += pcl::distances::l1(query, match);
				l2_sum += pcl::distances::l2(query, match);
				l2Sqr_sum += pcl::distances::l2Sqr(query, match);
			}
			l1_sum /= correspondences->size();
			l2_sum /= correspondences->size();
			l2Sqr_sum /= correspondences->size();
			return l2Sqr_sum;*/
			//}
			//get number of correspondences
		size_t cnt = correspondences->size();
		return (float)cnt;
	}

	//-------------------------------------------------------------------
	//shifting/roll in defined intervals with summing up correspondences
	//-------------------------------------------------------------------
	void shift_and_roll(float angle_min, float angle_max, float angle_step,
		float shift_min, float shift_max, float shift_step,
		std::vector<std::pair<float, float>>& angle_count, std::vector<std::pair<float, float>>& shift_and_count,
		Eigen::Matrix3f rotation, Eigen::Vector3f initialTranslation, Eigen::Vector3f direction,
		pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized, pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr, bool use_improvement) {
		int angle_max_index = 0;
		int angle_index = 0;
		for (float i = angle_min; i <= angle_max; i += angle_step) {
			float angleCount = 0.0f;
			for (float j = shift_min; j <= shift_max; j += shift_step) {
				Eigen::Matrix3f rot = rotateByAngle((float)i, rotation);
				Eigen::Vector3f trans = shiftByValue((float)j, initialTranslation, direction);
				Eigen::Matrix4f transform = buildTransformationMatrix(rot, trans);
				float correspondence_count = computeCorrespondences(transform, model_voxelized, point_cloud_ptr);
				std::vector<std::pair<float, float>>::iterator it;
				it = std::find_if(shift_and_count.begin(), shift_and_count.end(), [j](const std::pair<float, float>& p1) {
					return p1.first == j; });
				if (it != shift_and_count.end()) {
					shift_and_count.at(std::distance(shift_and_count.begin(), it)).second += correspondence_count;
				}
				else {
					shift_and_count.push_back(std::pair<float, float>(j, correspondence_count));
				}
				angleCount += correspondence_count;
			}
			//----------------------------------
			//algorithm performance improvement
			//if correspondence number didn't grow for 5 angles, stop everything because it won't get better
			if (use_improvement) {
				if (angle_max_index + ((std::abs(angle_min) + std::abs(angle_max)) / angle_step) / 4 == angle_index) {
					break;
				}
				//a better value than the current one was found, save the new index of the maximum value
				if (angle_index > angle_max_index && angleCount > angle_count.at(angle_max_index).second) {
					angle_max_index = angle_index;
				}
				angle_index++;
			}
			//----------------------------------
			angle_count.push_back(std::pair<float, float>(i, angleCount));
		}
	}

	//-------------------------------------------------------------------
	//shifting/roll in defined intervals without summing up correspondences
	//-------------------------------------------------------------------
	void shift_and_roll_without_sum(float angle_min, float angle_max, float angle_step,
		float shift_min, float shift_max, float shift_step,
		std::vector<std::tuple<float, float, float>>& count,
		Eigen::Matrix3f rotation, Eigen::Vector3f initialTranslation, Eigen::Vector3f direction,
		pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized, pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr) {
		int num_angle_steps = (angle_max - angle_min) / angle_step;
		int num_shift_steps = (shift_max - shift_min) / shift_step;
		for (int i = 0; i <= num_angle_steps * num_shift_steps + 1; i++) {
			count.push_back(std::tuple<float, float, float>(0, 0, 0));
		}
#pragma omp parallel for num_threads(omp_get_num_procs())
		for (int angle = 0; angle < num_angle_steps; angle++) {
			int shift_index = 0;
			for (float shift = shift_min; shift <= shift_max; shift += shift_step) {
				Eigen::Matrix3f rot = rotateByAngle((float)angle_min + angle * angle_step, rotation);
				Eigen::Vector3f trans = shiftByValue((float)shift, initialTranslation, direction);
				Eigen::Matrix4f transform = buildTransformationMatrix(rot, trans);
				float correspondence_count = computeCorrespondences(transform, model_voxelized, point_cloud_ptr);
				count.at(angle * num_shift_steps + shift_index) = std::tuple<float, float, float>(angle_min + angle * angle_step, shift, correspondence_count);
				shift_index++;
			}
		}
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
				std::vector<cv::Point> elipsePoints;
				MatToPointXYZ(std::get<2>(tup), std::get<3>(tup), elipsePoints, w, point_cloud_ptr, imageGray.rows, imageGray.cols);

				//compute center point of needle frame for translation
				if (elipsePoints.size() >= 50) { //to remove outliers, NOT RANSAC
					cv::RotatedRect elipse = cv::fitEllipse(cv::Mat(elipsePoints));
					pcl::PointXYZ peak;
					generatePoint(peak, elipse.center.x, elipse.center.y, w, imageGray.cols, imageGray.rows);
					peak_points->push_back(peak);
				}
			}
		}

		//downsample pointcloud
		float VOXEL_SIZE_ICP_ = 0.02f;
		pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_icp;
		voxel_grid_icp.setInputCloud(point_cloud_ptr);
		voxel_grid_icp.setLeafSize(VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
		voxel_grid_icp.filter(*point_cloud_ptr);

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
		pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_cut(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr peak_points(new pcl::PointCloud<pcl::PointXYZ>);
		boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>> needle_width = recognizeOCT(point_cloud_cut, peak_points, oct_dir, only_tip);
		pcl::PointCloud<pcl::PointXYZ>::Ptr model_half_Z(new pcl::PointCloud<pcl::PointXYZ>());
		cutModelinHalf(point_cloud_cut, point_cloud_ptr, 2);

		//-------------------------------
		//shifting algorithm
		//-------------------------------
		if (shift) {

			//-------------------------------
			//process the CAD mdoel
			//-------------------------------
			pcl::PointCloud<pcl::PointXYZ>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized_cut(new pcl::PointCloud<pcl::PointXYZ>());
			generatePointCloudFromModel(modelCloud, model_voxelized, path);
			int missing_frames = NUM_FRAMES - needle_width->size();
			cutPartOfModel(model_voxelized, model_voxelized_cut, 0);//missing_frames);
			/*pcl::PointCloud<pcl::PointXYZ>::Ptr model_half_Y(new pcl::PointCloud<pcl::PointXYZ>());
			cutModelinHalf(model_voxelized, model_half_Y, 1);
			pcl::PointCloud<pcl::PointXYZ>::Ptr model_half_Z(new pcl::PointCloud<pcl::PointXYZ>());
			cutModelinHalf(model_half_Y, model_half_Z, 2);*/


			//--------------------------------------
			//compute initial translation/rotation
			//--------------------------------------
			//compute the 3d direction of the needle
			std::pair<Eigen::Vector3f, Eigen::Vector3f> direction = computeNeedleDirection(peak_points);
			std::cout << "origin: " << std::endl << std::get<0>(direction) << std::endl << "direction: " << std::endl << std::get<1>(direction) << std::endl;

			//compute the 3d rotation of the needle
			Eigen::Matrix3f rotation = computeNeedleRotation(direction);
			std::cout << "rotation matrix: " << std::endl << rotation << std::endl;
			//rotate back to 0 degree on z axis
			Eigen::Vector3f euler = rotation.eulerAngles(0, 1, 2) * 180 / M_PI;
			rotation = rotateByAngle(180 - euler.z(), rotation);
			std::cout << "euler angles: " << std::endl << rotation.eulerAngles(0, 1, 2) * 180 / M_PI << std::endl;

			//compute 3d translation of the needle
			float tangencyPoint = regression(needle_width) / (float)NUM_FRAMES * SCALE_Z; //scaling
			std::cout << "tangency point: " << tangencyPoint << std::endl;
			Eigen::Vector3f initialTranslation = computeNeedleTranslation(tangencyPoint, std::get<0>(direction), std::get<1>(direction), getModelSize(model_voxelized) / 2);
			std::cout << "translation: " << std::endl << initialTranslation << std::endl;

			//build the transformation matrix with currently computed rotation and translation
			Eigen::Matrix4f transformation = buildTransformationMatrix(rotation, initialTranslation);

			//std::vector<std::pair<float, float>> angle_count;
			//std::vector<std::pair<float, float>> shift_count;
			std::vector<std::tuple<float, float, float>> correspondence_count;

			//--------------------------------------
			//start of shifting algorithm
			//--------------------------------------
			//initialize interval values
			float angleStart = -90.0f;
			float angleEnd = 90.0f;
			float angleStep = 1.0f;
			float shiftStart = 0.0f;
			float shiftEnd = 0.3f;
			float shiftStep = 0.05f;
			//int max_index_angles = 0;
			//int max_index_shift = 0;
			int correspondence_index = 0;
			float max_angle = 0.0f;
			float max_shift = 0.0f;

			bool use_improvement = false;
			int NUM_STEPS = 2;
			{
				pcl::ScopeTime t("Shift and Roll");
				for (int i = 0; i < 4; i++) {
					//angle_count.clear();
					//shift_count.clear();
					correspondence_count.clear();
					//apply shift and roll in small steps in given intervals and compute correspondences
					//shift_and_roll(angleStart, angleEnd, angleStep, shiftStart, shiftEnd, shiftStep, angle_count, shift_count, rotation, initialTranslation, std::get<1>(direction), model_voxelized, point_cloud_ptr, use_improvement);
					shift_and_roll_without_sum(angleStart, angleEnd, angleStep, shiftStart, shiftEnd, shiftStep, correspondence_count, rotation, initialTranslation, std::get<1>(direction), model_voxelized_cut, point_cloud_ptr);
					//find index of maximum correspondences
					//max_index_angles = findMaxIndexOfVectorOfPairs(angle_count);
					//max_index_shift = findMaxIndexOfVectorOfPairs(shift_count);
					correspondence_index = findMaxIndexOfVectorOfTuples(correspondence_count);
					max_angle = std::get<0>(correspondence_count.at(correspondence_index));
					max_shift = std::get<1>(correspondence_count.at(correspondence_index));

					//check bounds of vectors to make sure that in both directions of max indices you can go as far as specified
					/*int angle_min = checkMinBoundsForVectorIndex(NUM_STEPS, max_index_angles);
					int angle_max = checkMaxBoundsForVectorIndex(NUM_STEPS, max_index_angles, angle_count.size());
					int shift_min = checkMinBoundsForVectorIndex(NUM_STEPS, max_index_shift);
					int shift_max = checkMaxBoundsForVectorIndex(NUM_STEPS, max_index_shift, shift_count.size());*/
					angleStart = checkMinBoundsForValue(max_angle, angleStart, angleStep);
					angleEnd = checkMaxBoundsForValue(max_angle, angleEnd, angleStep);
					shiftStart = checkMinBoundsForValue(max_shift, shiftStart, shiftStep);
					shiftEnd = checkMaxBoundsForValue(max_shift, shiftEnd, shiftStep);

					//assign new interval values
					/*angleStart = angle_count.at(max_index_angles - angle_min).first;
					angleEnd = angle_count.at(max_index_angles + angle_max).first;
					shiftStart = shift_count.at(max_index_shift - shift_min).first;
					shiftEnd = shift_count.at(max_index_shift + shift_max).first;*/
					angleStep /= 5.0f;
					shiftStep /= 5.0f;
					//std::cout << "angle: " << angle_count.at(max_index_angles).first * -1 << std::endl;
					//std::cout << "shift: " << shift_count.at(max_index_shift).first << std::endl;
					std::cout << "angle: " << max_angle * -1 << std::endl;
					std::cout << "shift: " << max_shift << std::endl;
					std::cout << "end of round: " << i << std::endl;

					//show correspondence count as graph
					/*std::vector<float> angle_corr;
					for (int j = 0; j < angle_count.size(); j++) {
						angle_corr.push_back(angle_count.at(j).second);
					}
					showFloatGraph("Angle Correspondences", &angle_corr[0], angle_corr.size(), 0);
					std::vector<float> shift_corr;
					for (int j = 0; j < shift_count.size(); j++) {
						shift_corr.push_back(shift_count.at(j).second);
					}
					showFloatGraph("Shift Correspondences", &shift_corr[0], shift_corr.size(), 0);*/
				}
			}

			//transform point cloud to currently best values
			pcl::PointCloud<pcl::PointXYZ>::Ptr modelTransformed(new pcl::PointCloud<pcl::PointXYZ>);
			//transformation = buildTransformationMatrix(rotateByAngle(angle_count.at(max_index_angles).first, rotation), shiftByValue(shift_count.at(max_index_shift).first, initialTranslation, std::get<1>(direction)));
			transformation = buildTransformationMatrix(rotateByAngle(max_angle, rotation), shiftByValue(max_shift, initialTranslation, std::get<1>(direction)));
			pcl::transformPointCloud(*model_voxelized_cut, *modelTransformed, transformation);

			float x_middle_OCT = computeMiddle(point_cloud_ptr, 0.0f);

			float z_min = getMinZValue(modelTransformed);
			float x_middle_model = computeMiddle(modelTransformed, z_min);

			Eigen::Vector3f OCT_point(x_middle_OCT, 0.0f, 0.0f);
			float x_in_direction = (OCT_point - (std::abs(0.0f - z_min) / std::get<1>(direction).z() * std::get<1>(direction))).x();
			if (max_angle > 0) {
				x_in_direction = x_middle_OCT;
			}
			{
				pcl::ScopeTime t("Final Adjustments");
				if (x_middle_model < x_in_direction) {
					while (x_middle_model < x_in_direction) {
						transformation = buildTransformationMatrix(rotateByAngle(-0.5f, transformation.block(0, 0, 3, 3)), shiftByValue(max_shift, initialTranslation, std::get<1>(direction)));
						pcl::transformPointCloud(*model_voxelized_cut, *modelTransformed, transformation);
						x_middle_model = computeMiddle(modelTransformed, getMinZValue(modelTransformed));
					}
				}
				else if (x_middle_model > x_in_direction) {
					while (x_middle_model > x_in_direction) {
						transformation = buildTransformationMatrix(rotateByAngle(0.5f, transformation.block(0, 0, 3, 3)), shiftByValue(max_shift, initialTranslation, std::get<1>(direction)));
						pcl::transformPointCloud(*model_voxelized_cut, *modelTransformed, transformation);
						x_middle_model = computeMiddle(modelTransformed, getMinZValue(modelTransformed));
					}
				}
			}



			Eigen::Matrix3f end_rot = transformation.block(0, 0, 3, 3);
			Eigen::Vector3f eulerAngles = end_rot.eulerAngles(0, 1, 2);
			eulerAngles *= 180 / M_PI;
			std::cout << eulerAngles << std::endl;
			float end_angle = 0.0f;
			if (eulerAngles.z() < 0) {
				end_angle = -180 - eulerAngles.z();
			}
			else {
				end_angle = 180 - eulerAngles.z();
			}

			std::cout << std::endl << "final angle: " << end_angle << std::endl;

			//--------------------------------
			//visualization
			//--------------------------------
			pcl::PointXYZ point2;
			point2.x = std::get<0>(direction).x() + 2 * std::get<1>(direction).x();
			point2.y = std::get<0>(direction).y() + 2 * std::get<1>(direction).y();
			point2.z = std::get<0>(direction).z() + 2 * std::get<1>(direction).z();
			pcl::PointXYZ point3;
			point3.x = std::get<0>(direction).x() - 2 * std::get<1>(direction).x();
			point3.y = std::get<0>(direction).y() - 2 * std::get<1>(direction).y();
			point3.z = std::get<0>(direction).z() - 2 * std::get<1>(direction).z();
			//show model
			boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
			viewer->setBackgroundColor(0, 0, 0);
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler(model_voxelized, 0, 0, 255);
			viewer->addPointCloud<pcl::PointXYZ>(model_voxelized, rgb_handler, "model");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler3(point_cloud_ptr, 0, 255, 0);
			viewer->addPointCloud<pcl::PointXYZ>(point_cloud_ptr, rgb_handler3, "oct cloud");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler4(peak_points, 255, 10, 10);
			viewer->addPointCloud<pcl::PointXYZ>(peak_points, rgb_handler4, "peak points");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler5(modelTransformed, 0, 255, 255);
			viewer->addPointCloud<pcl::PointXYZ>(modelTransformed, rgb_handler5, "model transformed");
			viewer->addLine(point2, point3, "line");
			//viewer->addLine(pcl::PointXYZ(x_middle_OCT, 0.5f, 0), pcl::PointXYZ(x_in_direction, 0.5f - 2 * std::get<1>(direction).y(), -2 * std::get<1>(direction).z()), "middleOCT");
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
