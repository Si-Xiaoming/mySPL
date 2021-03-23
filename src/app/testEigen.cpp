#include<iostream>
#include<Eigen/Dense>
#include"nanoflann.hpp"
#include <ctime>
#include <cstdlib>
#include"testLaszip.h"

using namespace std;
using namespace nanoflann;

/*
只需要更改generateRandomPointCloud
*/


template <typename T>
void generateRandomPointCloud(PointCloud<T> &point, const size_t N, const T max_range = 10)
{
	// Generating Random Point Cloud
	point.pts.resize(N);
	for (size_t i = 0; i < N; i++)
	{
		point.pts[i].x = max_range * (rand() % 1000) / T(1000);
		point.pts[i].y = max_range * (rand() % 1000) / T(1000);
		point.pts[i].z = max_range * (rand() % 1000) / T(1000);
	}
}

// This is an exampleof a custom data set class


template <typename num_t>
void kdtree_demo(const size_t N, num_t x, num_t y, num_t z)
{
	PointCloud<num_t> cloud;

	// Generate points:
	generateRandomPointCloud(cloud, N);

	// construct a kd-tree index:
	typedef KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<num_t, PointCloud<num_t> >,
		PointCloud<num_t>,
		3 /* dim */
	> my_kd_tree_t;

	my_kd_tree_t   index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index.buildIndex();

	num_t query_pt[3];
	query_pt[0] = x;
	query_pt[1] = y;
	query_pt[2] = z;

	// ----------------------------------------------------------------
	// radiusSearch(): Perform a search for the points within search_radius
	// ----------------------------------------------------------------
	{
		const num_t search_radius = static_cast<num_t>(0.1);
		std::vector<std::pair<size_t, num_t> >   ret_matches;

		nanoflann::SearchParams params;
		params.sorted = true;

		const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches, params);

		cout << "radiusSearch(): radius=" << search_radius << " -> " << nMatches << " matches\n";
		for (size_t i = 0; i < nMatches; i++)
			cout << "idx[" << i << "]=" << ret_matches[i].first << " dist[" << i << "]=" << ret_matches[i].second << endl;
		cout << "\n";

		for (size_t i = 0; i < nMatches; i++) {
			cout << "第" << i << "个点：" << cloud.pts[ret_matches[i].first].x << "  " << cloud.pts[ret_matches[i].first].y << "  " << cloud.pts[ret_matches[i].first].z << endl;

		}
		cout << query_pt[1];
	}



}

//int main()
//{
//	// Randomize Seed
//	srand(static_cast<unsigned int>(time(nullptr)));
//	kdtree_demo<double>(100000, 0.5, 0.3, 0.5);
//	return 0;
//}