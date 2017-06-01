global_classification.cpp
	main: start creating views, setup global crh pipeline
	recognizeOCT: generate the oct point cloud and start computation of alignments
crh_estimator.h
	computation of camera roll histograms
cvfh_estimator.h
	computation of clustered viewpoint feature histograms
util.h
	counting files in directory, processing of paths
global_nn_recognizer_crh.h
	header for crh pipeline
global_nn_recognizer_crh.hpp
	initialize: computation of descriptors, create flann index
	recognize: start nearest neighbour search, perform camera roll histogram alignment, perform ICP
render_view_tesselated_sphere.cpp
	create the different views from a circle around the point cloud