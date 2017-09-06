#pragma once


#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <armadillo>
#include "opencv2\opencv.hpp"
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>


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

std::pair<Eigen::Vector3f, Eigen::Vector3f> fitLine(std::vector<Eigen::Vector3f>& points) {
	// copy coordinates to  matrix in Eigen format
	size_t num_atoms = points.size();
	Eigen::Matrix< Eigen::Vector3f::Scalar, Eigen::Dynamic, Eigen::Dynamic > centers(num_atoms, 3);
	for (size_t i = 0; i < num_atoms; ++i) centers.row(i) = points[i];

	Eigen::Vector3f origin = centers.colwise().mean();
	Eigen::MatrixXf centered = centers.rowwise() - origin.transpose();
	Eigen::MatrixXf cov = centered.adjoint() * centered;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);
	Eigen::Vector3f axis = eig.eigenvectors().col(2).normalized();
	//multiply with -1 so that it points towards origin
	return std::make_pair(origin, axis * -1.0f);
}

std::vector<int> getInliers(pcl::PointCloud<pcl::PointXYZ>::Ptr& peak_points) {
	std::vector<int> inliers;
	pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr
		model(new pcl::SampleConsensusModelLine<pcl::PointXYZ>(peak_points));
	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model);
	ransac.setDistanceThreshold(.1f);
	ransac.computeModel();
	ransac.getInliers(inliers);
	return inliers;
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