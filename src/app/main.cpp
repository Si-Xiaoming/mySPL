#include<iostream>
#include<Eigen/Dense>
#include <fstream>
#include "testRanger.h"
#include"testLaszip.h"
#include"testNanoflann.h"
void ReadData(std::istream &fin, Eigen::MatrixXf &m_matrix)
{
	int numRow = m_matrix.rows();
	int numCol = m_matrix.cols();

	Eigen::VectorXd vecPerRow(numRow);
	for (int j = 0; j < numRow; j++)//共numRow行
	{
		for (int i = 0; i < numCol; i++)//共numCol列组成一行
		{
			fin >> m_matrix(j, i);
		}

	}
}

void operate(PointCloud<double> &pc, PointCloud<double> &testPc) {
	

	MatrixXd mxf;
	FlannFeatureValue ffv(pc);
	int radius = 2;
	MatrixXf featureMatrix = ffv.kdTreeBuild(radius);//当半径为radius时，输出特征矩阵
	cout << "训练集特征矩阵计算完毕" << endl;

	/*写文件**/
	std::ofstream fout("trainFeatureMatrix.bin", std::ios::binary);
	fout << featureMatrix << std::endl;
	fout.flush();
	cout << "训练集特征矩阵写入完毕" << endl;


	MatrixXf testMxf;
	FlannFeatureValue testFfv(testPc);
	MatrixXf testFeatureMatrix = testFfv.kdTreeBuild(radius);
	cout << "测试集特征矩阵计算完毕" << endl;

	/*写文件**/
	std::ofstream fout2("testFeatureMatrix.bin", std::ios::binary);
	fout2 << testFeatureMatrix << std::endl;
	fout2.flush();
	cout << "测试集特征矩阵写入完毕" << endl;


	
}

int RandomForest(PointCloud<double> &pc, PointCloud<double> &testPc, LasOperate &testLo) {
	SuperpixelClassifier sc;
	VectorXf label = VectorXf::Zero(pc.kdtree_get_point_count());

	for (size_t i = 0; i < pc.kdtree_get_point_count(); i++) {
		if (pc.pts[i].classifcation == 7)
			label[i] = 1;//若不是类别7，则非噪声点
	}


	//读文件
	std::ifstream fin("trainFeatureMatrix.bin", std::ios::binary);
	if (!fin)
	{
		return 0;
	}
	MatrixXf featureMatrix = MatrixXf::Zero(pc.kdtree_get_point_count(), 3);
	ReadData(fin, featureMatrix);


	sc.train(featureMatrix, label);
	cout << "训练集随机森林计算完毕" << endl;


	//读文件
	std::ifstream fin2("testFeatureMatrix.bin", std::ios::binary);
	if (!fin2)
	{
		return 0;
	}
	MatrixXf testFeatureMatrix = MatrixXf::Zero(testPc.kdtree_get_point_count(), 3);
	ReadData(fin2, testFeatureMatrix);

	VectorXf label_out = sc.predict(testFeatureMatrix);
	int te = 0;
	for (size_t i = 0; i < testPc.kdtree_get_point_count(); i++) {
		if (label_out[i] == 1)
		{
			testPc.pts[i].classifcation = 7;
			te++;
		}
	}
	cout << te << endl;
	cout << "测试集分类完毕" << endl;
	string outFileName = "D:\\SPL100_noise\\data\\test\\outTest03.laz";

	testLo.MyPcSave(outFileName, testPc);
	cout << "数据写入文件完毕" << endl;
}

int main(int argc, char** argv) {

	


	string filename = "D:\\SPL100_noise\\data\\test\\train_Cloud.laz";

	string testFilename = "D:\\SPL100_noise\\data\\test\\test_Cloud.laz";

	LasOperate lo(filename);
	PointCloud<double> pc;
	pc = lo.pointRead();
	cout << "训练集数据读取完毕" << endl;


	LasOperate testLo(testFilename);
	PointCloud<double> testPc;
	testPc = testLo.pointRead();
	cout << "测试集数据读取完毕" << endl;
	//operate(pc,testPc);
	RandomForest(pc, testPc, testLo);


	//LasOperate lzo(filename);
	//cout << lzo.path << endl;
	//PointCloud<double> pc;
	//pc=lzo.pointRead();
	//cout <<pc.kdtree_get_point_count();
	//lzo.PcSave("D:\\myDataBase\\pointCloudDatas\\testlasDir\\test\\Cloudsss.laz", pc);



}
