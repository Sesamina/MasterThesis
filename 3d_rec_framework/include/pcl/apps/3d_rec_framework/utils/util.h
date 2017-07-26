#pragma once
#include <string>
#include <algorithm>
#include <Windows.h>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

// process the path to get the right format 
std::string getDirectoryPath(std::string path) {
	std::replace(path.begin(), path.end(), '\\', '/');
	int lastSlashIndex = path.find_last_of('/', (int)path.size());
	if (lastSlashIndex < (int)path.size() - 1)
		path += "/";
	return path;
}



int countNumberOfFilesInDirectory(std::string inputDirectory, const char* fileExtension) {
	char search_path[300];
	WIN32_FIND_DATA fd;
	sprintf_s(search_path, fileExtension, inputDirectory.c_str());
	HANDLE hFind = ::FindFirstFile(search_path, &fd);

	//count the number of OCT frames in the folder
	int count = 0;
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{

				count++;

			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return count;
}

void getModelsInDirectory(bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext) {
	bf::directory_iterator end_itr;
	for (bf::directory_iterator itr(dir); itr != end_itr; ++itr) {
		//check that it is a ply file and then add, otherwise ignore..
		std::vector < std::string > strs;
#if BOOST_FILESYSTEM_VERSION == 3
		std::string file = (itr->path().filename()).string();
#else
		std::string file = (itr->path()).filename();
#endif

		boost::split(strs, file, boost::is_any_of("."));
		std::string extension = strs[strs.size() - 1];

		if (extension.compare(ext) == 0)
		{
#if BOOST_FILESYSTEM_VERSION == 3
			std::string path = rel_path_so_far + (itr->path().filename()).string();
#else
			std::string path = rel_path_so_far + (itr->path()).filename();
#endif

			relative_paths.push_back(path);
		}
	}
}

void generatePointCloudFromModel(pcl::PointCloud<pcl::PointXYZ>::Ptr& modelCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& model_voxelized, std::string path) {
	//get models in directory
	std::vector < std::string > files;
	std::string start = "";
	std::string ext = std::string("ply");
	bf::path dir = path;
	getModelsInDirectory(dir, start, files, ext);
	std::stringstream model_path;
	model_path << path << "/" << files[0];
	std::string path_model = model_path.str();
	//sample points on surface of model
	pcl::rec_3d_framework::uniform_sampling(path_model, 100000, *modelCloud, 1.f);
	//downsample points
	float VOXEL_SIZE_ICP_ = 0.02f;
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_icp;
	voxel_grid_icp.setInputCloud(modelCloud);
	voxel_grid_icp.setLeafSize(VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
	voxel_grid_icp.filter(*model_voxelized);

	Eigen::Matrix4f rotationZ;
	rotationZ << 0, 1, 0, 0,
		-1, 0, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;
	pcl::transformPointCloud(*modelCloud, *modelCloud, rotationZ);
	pcl::transformPointCloud(*model_voxelized, *model_voxelized, rotationZ);
}

float getModelSize(pcl::PointCloud<pcl::PointXYZ>::Ptr modelCloud) {
	float min = modelCloud->points.at(0).z;
	float max = modelCloud->points.at(0).z;
	for (int i = 0; i < modelCloud->points.size(); i++) {
		if (modelCloud->points.at(i).z < min) {
			min = modelCloud->points.at(i).z;
		}
		else if (modelCloud->points.at(i).z > max) {
			max = modelCloud->points.at(i).z;
		}
	}
	float modelSize = std::abs(max - min);
	return modelSize;
}