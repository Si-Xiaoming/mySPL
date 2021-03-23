#include"testNanoflann.h"

typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3> my_kd_tree_t;
MatrixXf FlannFeatureValue::kdTreeBuild(double radius)
{
	my_kd_tree_t index(3 /*dim*/, points, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index.buildIndex();
	int column = 3;
	MatrixXf featureMatrix = MatrixXf::Zero(points.kdtree_get_point_count(),column);
	for (int i = 0; i < points.kdtree_get_point_count(); i++) {
		double query_pt[3];
		query_pt[0] = points.pts[i].x;
		query_pt[1] = points.pts[i].y;
		query_pt[2] = points.pts[i].z;

		{
			const double search_radius = static_cast<double>(radius);
			std::vector<std::pair<size_t, double> >   ret_matches;

			nanoflann::SearchParams params;
			params.sorted = false;

			const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches, params);

			MatrixXf mxf;
			mxf = MatrixXf::Zero(nMatches, 3);
			for (int i = 0; i < nMatches; i++) {
				mxf(i, 0) = points.pts[ret_matches[i].first].x;
				mxf(i, 1) = points.pts[ret_matches[i].first].y;
				mxf(i, 2) = points.pts[ret_matches[i].first].z;
			}
			VectorXf featureValues;
			MatrixXf featureVectors;

			CaculateFeatureValue(mxf, featureValues, featureVectors);
			//存储顺序按照 L  P  S  V
			//cout << " 返回的特征值" << featureValues << endl;

			if (featureValues(0) == 0) {
				featureValues(0) += 1e-17;
			}

			featureMatrix(i, 0) = (featureValues(0) - featureValues(1)) / featureValues(0);
			featureMatrix(i, 1) = (featureValues(1) - featureValues(2)) / featureValues(0);
			featureMatrix(i, 2) = featureValues(2) / featureValues(0);
		}
	}
	return featureMatrix;
}
//建立KD树
//MatrixXd FlannFeatureValue::buildRadiusKD(double x, double y, double z,double radius)
//{
//	// construct a kd-tree index:
//	typedef KDTreeSingleIndexAdaptor<
//		L2_Simple_Adaptor<double, PointCloud<double> >,
//		PointCloud<double>,
//		3 /* dim */
//	> my_kd_tree_t;
//
//	my_kd_tree_t   index(3 /*dim*/, points, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
//	index.buildIndex();
//
//	double query_pt[3];
//	query_pt[0] = x;
//	query_pt[1] = y;
//	query_pt[2] = z;
//
//	// ----------------------------------------------------------------
//	// radiusSearch(): Perform a search for the points within search_radius
//	// ----------------------------------------------------------------
//	{
//		const double search_radius = static_cast<double>(radius);
//		std::vector<std::pair<size_t, double> >   ret_matches;
//
//		nanoflann::SearchParams params;
//		params.sorted = false;
//
//		const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches, params);
//
//		MatrixXd mxf;
//		mxf = MatrixXd::Zero(nMatches, 3);
//		for (int i = 0; i < nMatches; i ++ ) {
//			mxf(i, 0) = points.pts[ret_matches[i].first].x;
//			mxf(i, 1) = points.pts[ret_matches[i].first].y;
//			mxf(i, 2) = points.pts[ret_matches[i].first].z;
//		}
//		return mxf;
//	}
//}
//将特征向量按照特征值大小排序
void FlannFeatureValue::sortEigenVectorByValues(eigVector & eigenValues, eigMatrix & eigenVectors)
{
	stdVectorTuple eigenValueAndVector;
	int size = static_cast<int>(eigenValues.size());

	eigenValueAndVector.reserve(size);
	for (int i = 0; i < size; ++i)
		eigenValueAndVector.push_back(stdTupleEigen(eigenValues[i], eigenVectors.col(i)));

	// 使用标准库中的sort，按从大到小排序
	std::sort(eigenValueAndVector.begin(), eigenValueAndVector.end(),
		[&](const stdTupleEigen& a, const stdTupleEigen& b) -> bool {
		return std::get<0>(a) > std::get<0>(b);
	});

	for (int i = 0; i < size; ++i) {
		eigenValues[i] = std::get<0>(eigenValueAndVector[i]); // 排序后的特征值
		eigenVectors.col(i).swap(std::get<1>(eigenValueAndVector[i])); // 排序后的特征向量
	}
}
//计算特征值
void FlannFeatureValue::CaculateFeatureValue(MatrixXf  mxd, VectorXf & featureValue, MatrixXf & featureVector)
{
	VectorXf meanVec = mxd.colwise().mean();
	MatrixXf meanmat = MatrixXf::Zero(mxd.rows(), mxd.cols());
	for (int i = 0; i < mxd.rows(); i++) {
		meanmat.row(i) = meanVec;
	}
	mxd -= meanmat;
	MatrixXf covMat = mxd.transpose()*mxd / mxd.rows();

	//cout << covMat << endl;


	EigenSolver<MatrixXf> solver(covMat);
	featureValue = solver.pseudoEigenvalueMatrix().diagonal();
	featureVector = solver.pseudoEigenvectors();
	sortEigenVectorByValues(featureValue, featureVector);
	//cout << "feature" << featureValue << endl;
}
//计算特征矩阵（  L P S V  ）
//MatrixXd FlannFeatureValue::CaculateFeatureMatrix(double radius)
//{
//	//有多少特征
//	int column = 3;
//
//	MatrixXd featureMatrix = MatrixXd::Zero(points.kdtree_get_point_count(), column);
//	
//
//	for (size_t i = 0; i < points.kdtree_get_point_count(); i++) {
//		MatrixXf bRKD;
//		VectorXf featureValues;
//		MatrixXf featureVectors;
//		bRKD = buildRadiusKD(points.pts[i].x, points.pts[i].y, points.pts[i].z, radius);
//		
//		CaculateFeatureValue(bRKD, featureValues, featureVectors);
//		//存储顺序按照 L  P  S  V
//		//cout <<" 返回的特征值" <<featureValues << endl;
//		featureMatrix(i, 0) = (featureValues(0) - featureValues(1)) / featureValues(0);
//		featureMatrix(i, 1) = (featureValues(1) - featureValues(2)) / featureValues(0);
//		featureMatrix(i, 2) = featureValues(3) / featureValues(1);
//	}
//	return featureMatrix;
//}
