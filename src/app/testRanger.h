#include<iostream>
#include<Eigen/Dense>
#include"nanoflann.hpp"
#include <ctime>
#include <cstdlib>
#include"testLaszip.h"
class Forest;
class Data;
class SuperpixelClassifier {
public:
	SuperpixelClassifier();
	~SuperpixelClassifier();

	// feature N x M,其中N 是特征个数，M是样本个数
	void train(const MatrixXf &features, const VectorXf &labels);
	VectorXf predict(const MatrixXf &features);

	void save(const std::string &path);
	void load(const std::string &path);

protected:
	std::shared_ptr<Forest> rf_;
};

class SuperpixelClassifierProbability {
public:
	SuperpixelClassifierProbability();
	~SuperpixelClassifierProbability();

	// feature N x M,其中N 是特征个数，M是样本个数
	void train(const MatrixXf &features, const VectorXf &labels);
	// 结果 是 K X M， 其中 K 是类别总数，M 是样本个数
	MatrixXf predict(const MatrixXf &features);

	void save(const std::string &path);
	void load(const std::string &path);

protected:
	std::shared_ptr<Forest> rf_;
};