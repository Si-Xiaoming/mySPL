#include<iostream>
#include<Eigen/Dense>
#include"nanoflann.hpp"
#include <ctime>
#include <cstdlib>
#include"testLaszip.h"

using namespace std;
using namespace nanoflann;


class FlannFeatureValue {
	using eigMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
	using eigVector = Eigen::Matrix<float, Eigen::Dynamic, 1>;
	using eigEigenSolver = Eigen::EigenSolver<eigMatrix>;
	using stdTupleEigen = std::tuple<float, eigMatrix>;
	using stdVectorTuple = std::vector<stdTupleEigen>;
	typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double,PointCloud<double>>,PointCloud<double>,3> my_kd_tree_t;
public:
	
	//my_kd_tree_t   index(3 /*dim*/, points, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	PointCloud<double> points;

	FlannFeatureValue(PointCloud<double> &points_) :points(points_) {
		my_kd_tree_t   index(3 /*dim*/, points, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	}
	~FlannFeatureValue() {}

	MatrixXf kdTreeBuild(double radius);

	MatrixXd buildRadiusKD(double x, double y, double z,double radius);

	void CaculateFeatureValue(MatrixXf mxd, VectorXf &featureValue, MatrixXf &featureVector);

	MatrixXd CaculateFeatureMatrix(double radius);

private:
	void sortEigenVectorByValues(eigVector& eigenValues, eigMatrix& eigenVectors);
};

