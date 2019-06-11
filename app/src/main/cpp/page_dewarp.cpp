
#include <math.h>
#include <algorithm>
#include "page_dewarp.hpp"
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <fstream>
#include "C_preprocess.hpp"

using namespace std;
using namespace cv;

#include "nlopt.hpp"

std::vector<cv::Point2d> span_points_flat;
std::vector<cv::Point2d> corners;
std::vector<int> keypoint_index[2];
std::vector<cv::Point2d> keypoint;
std::vector<double> params, out_params, out_page_dims;
double dims[2];
std::vector<cv::Scalar> colors_debug {	cv::Scalar(255, 0, 0),
                                       	cv::Scalar(255, 63, 0),
                                        cv::Scalar(255, 127, 0),
                                        cv::Scalar(255, 191, 0),
                                        cv::Scalar(255, 255, 0),
                                        cv::Scalar(191, 255, 0),
                                        cv::Scalar(127, 255, 0),
                                        cv::Scalar(63, 255, 0),
                                        cv::Scalar(0, 255, 0),
                                        cv::Scalar(0, 255, 63),
                                        cv::Scalar(0, 255, 127),
                                        cv::Scalar(0, 255, 191),
                                        cv::Scalar(0, 255, 255),
                                        cv::Scalar(0, 191, 255),
                                        cv::Scalar(0, 127, 255),
                                        cv::Scalar(0, 63, 255),
                                        cv::Scalar(0, 0, 255),
                                        cv::Scalar(63, 0, 255),
                                        cv::Scalar(127, 0, 255),
                                        cv::Scalar(191, 0, 255),
                                        cv::Scalar(255, 0, 255),
                                        cv::Scalar(255, 0, 191),
                                        cv::Scalar(255, 0, 127),
                                        cv::Scalar(255, 0, 63)
                                      };
bool sort_contourInfo (const ContourInfo ci0, const ContourInfo ci1)
    {
        return ci0.rect.y < ci1.rect.y;
}

std::vector<cv::Point2d> norm2pix(cv::Size s, std::vector<cv::Point2d> pts, bool as_integer)
{
	double height = s.height;
	double width = s.width;
	unsigned int i;
	std::vector<cv::Point2d> pts_out(pts.size());
	double scl = std::max(width, height) * 0.5;
	cv::Point offset(width * 0.5, height * 0.5);
	for (i = 0; i < pts.size(); ++i)
	{
		pts_out[i].x = pts[i].x * scl + offset.x;
		pts_out[i].y = pts[i].y * scl + offset.y;
		if (as_integer)
		{
			pts[i].x = (double)(pts[i].x + 0.5);
			pts[i].y = (double)(pts[i].y + 0.5);
		}
	}
	//pts_out = pts;
	return pts_out;
}
void visualize_spans(cv::Mat small, cv::Mat pagemask, std::vector<std::vector<ContourInfo>> spans)
{
	cv::Mat drawing = cv::Mat::zeros(small.size(), CV_8UC3);
	std::vector<std::vector<Point>> contours_debug;
	cv::RNG rng(12345);
	for (unsigned int i = 0; i < spans.size(); i++)
	{
		for (unsigned int j = 0; j < spans[i].size(); j++)
			contours_debug.push_back(spans[i][j].contour);
	}
	for (unsigned int i = 0; i < contours_debug.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		cv::drawContours(drawing, contours_debug, i, color, 2, 8, std::vector<cv::Vec4i>(), 0, Point());
	}
	cv::Mat matD;
	cv::add(small, drawing, matD);
	//printf("contours size = %lu\n", contours.size());
	if (DEBUG_LEVEL)
		cv::imwrite("spans.png", matD);
	//exit(1);
}

void visualize_span_points(cv::Mat small, std::vector<std::vector<cv::Point2d>> span_points,
						   std::vector<cv::Point2d> corners)
{
	cv::Mat display = small.clone();
	std::vector<cv::Point2d> points;
	cv::Point2d mean, point0, point1;
	std::vector<Point2d> eigen_vecs(1);
	cv::Mat dps;
	double dpm, min, max;
	for (unsigned int i = 0; i < span_points.size(); i++)
	{
		points = norm2pix(small.size(), span_points[i], false);
		cv::Mat data_pts(span_points[i].size(), 2, CV_64FC1);

		for (unsigned int j = 0; j < data_pts.size().height; ++j)
		{
			data_pts.at<double>(j, 0) = points[j].x;
			data_pts.at<double>(j, 1) = points[j].y;
		}

		cv::PCA pca_analysis(data_pts, Mat(), 0);
		mean = cv::Point2d(pca_analysis.mean.at<double>(0, 0),
						   pca_analysis.mean.at<double>(0, 1));
		eigen_vecs[0] = cv::Point2d(pca_analysis.eigenvectors.at<double>(0, 0),
									pca_analysis.eigenvectors.at<double>(0, 1));
		dps = cv::Mat(points) * cv::Mat(eigen_vecs);
		dpm = mean.x * eigen_vecs[0].x + mean.y * eigen_vecs[0].y;
		cv::minMaxLoc(dps, &min, &max);
		point0 = mean + eigen_vecs[0] * (min - dpm);
		point1 = mean + eigen_vecs[0] * (max - dpm);
		for (unsigned int j = 0; j < points.size(); j++)
		{
			cv::circle(display, cv::Point2d(points[j].x, points[j].y), 3,
					   colors_debug[i % colors_debug.size()], -1, cv::LINE_AA);
			//printf("NUMBER OF POINT = %lu\n", points.size());
		}

		cv::line(display, point0, point1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
	}
	std::vector<cv::Point2d> point_lines = norm2pix(small.size(), corners, true);
	std::vector<cv::Point> point_convert;
	for (unsigned int i = 0; i < point_lines.size(); i++)
	{
		point_convert.push_back(cv::Point((int)point_lines[i].x, (int)point_lines[i].y));
	}
	cv::polylines(display, point_convert,
				  true, cv::Scalar(255, 255, 255));
	if (DEBUG_LEVEL)
		cv::imwrite("span_point.png", display);
}

std::vector<cv::Point2d> pix2norm(cv::Size s, std::vector<cv::Point2d> pts)
{
	std::vector<cv::Point2d> pts_out(pts.size());
	double height = s.height;
	double width = s.width;
	unsigned int i;
	double scl = 2 / std::max(width, height);
	cv::Point2d offset(width * 0.5, height * 0.5);
	for (i = 0; i < pts.size(); ++i)
	{
		pts_out[i].x = (pts[i].x - offset.x) * scl;
		pts_out[i].y = (pts[i].y - offset.y) * scl;
	}
	return pts_out;
}

void draw_correspondences(string filename, cv::Mat img, std::vector<cv::Point2d> dstpoints, std::vector<cv::Point2d> projpts)
{
	cv::Mat display = img.clone();
	dstpoints = norm2pix(img.size(), dstpoints, true);
	projpts = norm2pix(img.size(), projpts, true);
	for (unsigned int i = 0; i < dstpoints.size(); i++)
	{
		cv::Point point((int)dstpoints[i].x, (int)dstpoints[i].y);
		cv::circle(display, point, 3, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
	}
	for (unsigned int i = 0; i < projpts.size(); i++)
	{
		cv::Point point((int)projpts[i].x, (int)projpts[i].y);
		cv::circle(display, point, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
	}
	
	for (unsigned int i = 0; i < dstpoints.size() - 1; i++)
	{
		printf("i = %u\n", i);
		printf("dstpoints.size() = %lu\n", dstpoints.size());
		cv::Point point1((int)dstpoints[i].x, (int)dstpoints[i].y);
		cv::Point point2((int)projpts[i].x, (int)projpts[i].y);
		cv::line(display, point1, point2,
				 cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
	}
	if (DEBUG_LEVEL)
		cv::imwrite(filename, display);
}

void blob_mean_and_tangent(std::vector<cv::Point> contour, double *center, double *tangent)
{
	cv::Moments moments = cv::moments(contour);
	double area = moments.m00;
	double area_inverse = 1 / area;
	double mean_x = moments.m10 * area_inverse;
	double mean_y = moments.m01 * area_inverse;

	cv::Mat moments_matrix(2, 2, CV_64F);
	moments_matrix.at<double>(0, 0) = moments.mu20 * area_inverse;
	moments_matrix.at<double>(0, 1) = moments.mu11 * area_inverse;
	moments_matrix.at<double>(1, 0) = moments_matrix.at<double>(0, 1);
	moments_matrix.at<double>(1, 1) = moments.mu02 * area_inverse;

	cv::Mat svd_u, svd_w, svd_vt;

	cv::SVDecomp(moments_matrix, svd_w, svd_u, svd_vt);
	center[0] = mean_x;
	center[1] = mean_y;

	tangent[0] = svd_u.at<double>(0, 0);
	tangent[1] = svd_u.at<double>(1, 0);
}

void get_mask(std::string name, cv::Mat small, cv::Mat page_mask, int mask_type, cv::Mat *mask)
{
	// int width = small.size().width;
	// int height = small.size().height;
	//printf("WIDTH - %d\n", width);
	// convert small to gray
	cv::Mat sgray;
	cv::cvtColor(small, sgray, cv::COLOR_RGB2GRAY);

	cv::Mat element;
	if (DEBUG_LEVEL)
		cv::imwrite("original_image.png", small);
	if (mask_type == MASK_TYPE_TEXT)
	{
		cv::adaptiveThreshold(sgray, *mask, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, ADAPTIVE_WINSZ, 25);
		if (DEBUG_LEVEL)
			cv::imwrite("threshold.png", *mask);
		element = cv::getStructuringElement(MORPH_RECT, cv::Size(9, 1));
		cv::dilate(*mask, *mask, element);
		if (DEBUG_LEVEL)
			cv::imwrite("dilate.png", *mask);
		element = cv::getStructuringElement(MORPH_RECT, cv::Size(1, 3));
		cv::erode(*mask, *mask, element);
		if (DEBUG_LEVEL)
			cv::imwrite("erode.png", *mask);
	}
	else
	{
		cv::adaptiveThreshold(sgray, *mask, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, ADAPTIVE_WINSZ, 7);
		if (DEBUG_LEVEL)
			cv::imwrite("threshold.png", *mask);
		element = cv::getStructuringElement(MORPH_RECT, cv::Size(3, 1));
		cv::erode(*mask, *mask, element, cv::Point(-1, -1), 3);
		if (DEBUG_LEVEL)
			cv::imwrite("dilate.png", *mask);
		element = cv::getStructuringElement(MORPH_RECT, cv::Size(8, 2));
		cv::dilate(*mask, *mask, element);
		if (DEBUG_LEVEL)
			cv::imwrite("erode.png", *mask);
	}
}

ContourInfo::ContourInfo(std::vector<cv::Point> c, cv::Rect r, cv::Mat m)
{
	this->contour = c;
	this->rect = r;
	this->mask = m;
	unsigned int i;
	blob_mean_and_tangent(this->contour, this->center, this->tangent);
	this->angle = atan2(this->tangent[1], this->tangent[0]);
	double clx[contour.size()];
	for (i = 0; i < contour.size(); ++i)
	{
		clx[i] = this->project_x(c[i]);
	}

	/* TODO: optimize */
	// this->local_xrng[0] = *std::min_element(clx, clx + contour.size());
	// this->local_xrng[1] = *std::max_element(clx, clx + contour.size());
	double min_lx = *std::min_element(clx, clx + contour.size());
	double max_lx = *std::max_element(clx, clx + contour.size());

	this->local_xrng[0] = min_lx;
	this->local_xrng[1] = max_lx;

	this->point0[0] = this->center[0] + this->tangent[0] * min_lx;
	this->point0[1] = this->center[1] + this->tangent[1] * min_lx;
	this->point1[0] = this->center[0] + this->tangent[0] * max_lx;
	this->point1[1] = this->center[1] + this->tangent[1] * max_lx;

	this->pred = NULL;
	this->succ = NULL;
}

double ContourInfo::interval_measure_overlap(double *int_a, double *int_b)
{
	return std::min(int_a[1], int_b[1]) - std::max(int_a[0], int_b[0]);
}


void resize_to_screen(cv::Mat src, cv::Mat *dst, int max_width = 1280, int max_height = 700)
{
	int width = src.size().width;
	int height = src.size().height;

	double scale_x = double(width) / max_width;
	double scale_y = double(height) / max_height;

	int scale = (int)ceil(scale_x > scale_y ? scale_x : scale_y);

	if (scale > 1)
	{
		double invert_scale = 1 / (double)scale;
		cv::resize(src, *dst, cv::Size(0, 0), invert_scale, invert_scale);
	}
	else
	{
		*dst = src.clone();
	}
}

// output: change page_mask and page_outline
void get_page_extents(cv::Mat small, std::vector <cv::Point2f> line_point, cv::Mat &page_mask, std::vector<cv::Point> *page_outline)
{
	int width = small.size().width;
	int height = small.size().height;

	int min_x = PAGE_MARGIN_X;
	int min_y = PAGE_MARGIN_Y;
	int max_x = width - PAGE_MARGIN_X;
	int max_y = height - PAGE_MARGIN_Y;

//	__android_log_print(ANDROID_LOG_ERROR, "DEBUG_get_page_extent", "%d", line_point.size());

	page_mask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
    if (line_point.size() < 4) {
        cv::rectangle(page_mask, cv::Point(min_x, min_y), cv::Point(max_x, max_y), cv::Scalar(255, 255, 255), -1);
        page_outline->push_back(cv::Point(min_x, min_y));
        page_outline->push_back(cv::Point(min_x, max_y));
         page_outline->push_back(cv::Point(max_x, max_y));
         page_outline->push_back(cv::Point(max_x, min_y));
    }

	else {
        cv::rectangle(page_mask, cv::Point((int)line_point[0].x, (int)line_point[0].y), cv::Point((int)line_point[3].x, (int)line_point[3].y), cv::Scalar(255, 255, 255), -1);
        page_outline->push_back(cv::Point((int)line_point[0].x, (int)line_point[0].y));
        page_outline->push_back(cv::Point((int)line_point[1].x, (int)line_point[1].y));
        page_outline->push_back(cv::Point((int)line_point[3].x, (int)line_point[3].y));
        page_outline->push_back(cv::Point((int)line_point[2].x, (int)line_point[2].y));
	}

	// page_outline->push_back(cv::Point(min_x, min_y));
	// page_outline->push_back(cv::Point(min_x, max_y));
	// page_outline->push_back(cv::Point(max_x, max_y));
	// page_outline->push_back(cv::Point(max_x, min_y));

}

void make_tight_mask(
	std::vector<cv::Point> contour,
	int min_x,
	int min_y,
	int width,
	int height,
	cv::Mat *tight_mask)
{
	unsigned int i;
	// each mask is a zeroes matrix whose width and height are equal to the width and height
	// of the bounding rect of each contour

	*tight_mask = cv::Mat::zeros(height, width, CV_8UC1);
	// std::vector <cv::Point> tight_contour;
	// std::vector<cv::Point> vect(contour.size(), cv::Point(min_x, min_y));
	// cv::Mat tight_contour = cv::Mat(contour) - cv::Mat(vect);

	std::vector<cv::Point> tight_contour(contour.size());
	for (i = 0; i < tight_contour.size(); ++i)
	{
		tight_contour[i].x = contour[i].x - min_x;
		tight_contour[i].y = contour[i].y - min_y;
	}

	// the tight contour is the original contour remove to the upper left corner
	// to fit into the tight_mask

	std::vector<std::vector<cv::Point>> tight_contours(1, tight_contour);

	cv::drawContours(*tight_mask, tight_contours, 0, Scalar(1, 1, 1), -1);
}

void get_contours_s(
	std::string name,
	cv::Mat small,
	cv::Mat page_mask,
	int mask_type,
	std::vector<ContourInfo> &contours_out)
{
	cv::Mat mask, hierarchy;
	unsigned int i;
	get_mask(name, small, page_mask, mask_type, &mask);
	// cv::namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	// cv::imshow( "Display window", mask );                   // Show our image inside it.
	// cv::waitKey(0);
	std::vector<std::vector<Point>> contours;
	std::vector<std::vector<Point>> contours_debug;
	cv::findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	//printf("contours.size = %lu\n", contours.size());
	//exit(1);
	cv::Rect rect;
	int min_x, min_y, width, height;
	cv::Mat tight_mask;

	for (i = 0; i < contours.size(); ++i)
	{
		rect = cv::boundingRect(contours[i]);
		min_x = rect.x;
		min_y = rect.y;
		width = rect.width;
		height = rect.height;
		if (width < TEXT_MIN_WIDTH || height < TEXT_MIN_HEIGHT || width < TEXT_MIN_ASPECT * height)
			continue;

		make_tight_mask(contours[i], min_x, min_y, width, height, &tight_mask);
		
		cv::Mat reduced_tight_mask;
		// cout << tight_mask;
		// printf("\n");
		cv::reduce(tight_mask, reduced_tight_mask, 0, cv::REDUCE_SUM, CV_32SC1);
		//cout << reduced_tight_mask;

		// reduced_tight_mask is a vector whose elements are respectively equal to
		// the height of the contours

		double max, min;
		 cv::minMaxLoc(reduced_tight_mask, &min, &max);
		// printf("min = %lf, max = %lf\n", min, max);
		//exit(1);
		// if the height of the heighest of the contour is greater that TEXT_MAX_THICKNESS
		// the contour is not considered as text

		if (max > TEXT_MAX_THICKNESS)
		{
			continue;
		}
		// each ContourInfo has: the contour itself, the boudding rect, and the tight_mask
		// aka a Mat with width and height equal to that of the bounding rect
		// and the Contour drawn on it
		contours_out.push_back(ContourInfo(contours[i], rect, tight_mask));
		contours_debug.push_back(contours[i]);
	}
	cv::Mat drawing = cv::Mat::zeros(mask.size(), CV_8UC3);
	cv::RNG rng(12345);
	//cout << contours;
	for (unsigned int i = 0; i < contours_debug.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		cv::drawContours(drawing, contours_debug, i, color, 2, 8, std::vector<cv::Vec4i>(), 0, Point());
	}
	printf("contours size = %lu\n", contours.size());
	if (DEBUG_LEVEL)
		cv::imwrite("contour.png", drawing);
	//exit(1);
	// TODO: if (DEBUG_LEVEL >= 2)
}

double angle_dist(double angle_b, double angle_a)
{
	double diff = angle_b - angle_a;
	while (diff > M_PI)
	{
		diff -= 2 * M_PI;
	}
	while (diff < -M_PI)
	{
		diff += 2 * M_PI;
	}
	return diff > 0 ? diff : -diff;
}

void generate_candidate_edge(
	ContourInfo *cinfo_a,
	ContourInfo *cinfo_b,
	Edge *var_Edge)
{
	int swap = 0;
	ContourInfo _cinfo_a(*cinfo_a);
	ContourInfo _cinfo_b(*cinfo_b);
	if (_cinfo_a.point0[0] > _cinfo_a.point0[0])
	{
		// cout << "DEBUG_GENERATE_CANDIDATE_EDGE_1:" << cinfo_a << "/" << cinfo_a->pred << "/" << cinfo_a->succ << "/" << cinfo_b
		// 	<< "/" << cinfo_b->pred << "/" << cinfo_b->succ << endl;
		swap = 1;
		ContourInfo temp(_cinfo_a);
		_cinfo_a = _cinfo_b;
		_cinfo_b = temp;
		// cout << "DEBUG_GENERATE_CANDIDATE_EDGE_2:" << cinfo_a << "/" << cinfo_a->pred << "/" << cinfo_a->succ << "/" << cinfo_b
		// 	<< "/" << cinfo_b->pred << "/" << cinfo_b->succ << endl;
	}

	double x_overlap_a = _cinfo_a.local_overlap(_cinfo_b);
	double x_overlap_b = _cinfo_b.local_overlap(_cinfo_a);

	double overall_tangent[2];
	overall_tangent[0] = _cinfo_b.center[0] - _cinfo_a.center[0];
	overall_tangent[1] = _cinfo_b.center[1] - _cinfo_a.center[1];

	double overall_angle = atan2(overall_tangent[1], overall_tangent[0]);

	double delta_angle = std::max(
							 angle_dist(_cinfo_a.angle, overall_angle),
							 angle_dist(_cinfo_b.angle, overall_angle)) *
						 180 / M_PI;

	double x_overlap = std::max(x_overlap_a, x_overlap_b);

	double dist = sqrt(
		pow(cinfo_b->point0[0] - cinfo_a->point1[0], 2) +
		pow(cinfo_b->point0[1] - cinfo_a->point1[1], 2));

	if (dist > EDGE_MAX_LENGTH || x_overlap > EDGE_MAX_OVERLAP || delta_angle > EDGE_MAX_ANGLE)
	{
		return;
	}
	else
	{
		double score = dist + delta_angle * EDGE_ANGLE_COST;
		// cout << "DEBUG_GENERATE_CANDIDATE_EDGE_3:" << score << "/" << cinfo_a->contour.size() << "/" << cinfo_b->contour.size() << endl;
		if (swap)
		{
			*var_Edge = Edge(score, cinfo_b, cinfo_a);
		}
		else
		{
			*var_Edge = Edge(score, cinfo_a, cinfo_b);
		}
	}
}

void assemble_spans(
	std::string name,
	cv::Mat small,
	cv::Mat page_mask,
	std::vector<ContourInfo> cinfo_list,
	std::vector<std::vector<ContourInfo>> *spans)
{
	unsigned int i, j;
	std::sort(cinfo_list.begin(), cinfo_list.end(), sort_contourInfo);

	std::vector<Edge> candidate_edges;

	for (i = 0; i < cinfo_list.size(); ++i)
	{
		for (j = 0; j < i; ++j)
		{
			Edge edge;
			generate_candidate_edge(&cinfo_list[i], &cinfo_list[j], &edge);
			if (edge.score)
			{
				candidate_edges.push_back(edge);
				// cout << "DEBUG_ASSEMBLE_SPANS_GENERATE_CANDIDATE_EDGE_1:" << cinfo_list[i].center[0] << "/" << cinfo_list[i].center[1] << "|" <<
				// 	cinfo_list[i].tangent[0] << "/" << cinfo_list[i].tangent[1] << "|" << cinfo_list[i].angle << "|" <<
				// 	cinfo_list[i].local_xrng[0] << "/" << cinfo_list[i].local_xrng[1] << "|" <<
				// 	cinfo_list[i].point0[0] << "/" << cinfo_list[i].point0[1] << "|" <<
				// 	cinfo_list[i].point1[0] << "/" << cinfo_list[i].point1[1] << endl;
				// cout << "DEBUG_ASSEMBLE_SPANS_GENERATE_CANDIDATE_EDGE_2:" << cinfo_list[j].center[0] << "/" << cinfo_list[j].center[1] << "|" <<
				// 	cinfo_list[j].tangent[0] << "/" << cinfo_list[j].tangent[1] << "|" << cinfo_list[j].angle << "|" <<
				// 	cinfo_list[j].local_xrng[0] << "/" << cinfo_list[j].local_xrng[1] << "|" <<
				// 	cinfo_list[j].point0[0] << "/" << cinfo_list[j].point0[1] << "|" <<
				// 	cinfo_list[j].point1[0] << "/" << cinfo_list[j].point1[1] << endl;
			}
		}
	}

	std::sort(candidate_edges.begin(), candidate_edges.end(), Edge::sortEdge);
	// for (int i = 0; i < candidate_edges.size(); ++i) {
	// 	cout << "DEBUG_ASSMEBLE_SPANS_GENERATE_CANDIDATE_EDGE_3:" <<
	// 		candidate_edges[i].cinfo_a->center[0] << "/" << candidate_edges[i].cinfo_a->center[1] << "|" <<
	// 		candidate_edges[i].cinfo_a->tangent[0] << "/" << candidate_edges[i].cinfo_a->tangent[1] << "|" << candidate_edges[i].cinfo_a->angle << "|" <<
	// 		candidate_edges[i].cinfo_a->local_xrng[0] << "/" << candidate_edges[i].cinfo_a->local_xrng[1] << "|" <<
	// 		candidate_edges[i].cinfo_a->point0[0] << "/" << candidate_edges[i].cinfo_a->point0[1] << "|" <<
	// 		candidate_edges[i].cinfo_a->point1[0] << "/" << candidate_edges[i].cinfo_a->point1[1] << endl;
	// 	cout << "DEBUG_ASSMEBLE_SPANS_GENERATE_CANDIDATE_EDGE_4:" <<
	// 		candidate_edges[i].cinfo_b->center[0] << "/" << candidate_edges[i].cinfo_b->center[1] << "|" <<
	// 		candidate_edges[i].cinfo_b->tangent[0] << "/" << candidate_edges[i].cinfo_b->tangent[1] << "|" << candidate_edges[i].cinfo_b->angle << "|" <<
	// 		candidate_edges[i].cinfo_b->local_xrng[0] << "/" << candidate_edges[i].cinfo_b->local_xrng[1] << "|" <<
	// 		candidate_edges[i].cinfo_b->point0[0] << "/" << candidate_edges[i].cinfo_b->point0[1] << "|" <<
	// 		candidate_edges[i].cinfo_b->point1[0] << "/" << candidate_edges[i].cinfo_b->point1[1] << endl;
	// 	cout << "DEBUG_ASSEMBLE_SPANS_GENERATE_CANDIDATE_EDGE_5:" << candidate_edges[i].score << endl;
	// }

	// cout << "DEBUG_ASSEMBLE_SPANS_3:" << candidate_edges.size() << endl;
	// cout << "DEBUG_ASSMEBLE_SPANS_4:";
	// for (int k = 0; k < candidate_edges.size(); ++k) {
	// 	cout << candidate_edges[k].score << "/" << candidate_edges[k].cinfo_a << "/" << candidate_edges[k].cinfo_b << endl;
	// }

	int p = 0;
	for (i = 0; i < candidate_edges.size(); ++i)
	{
		if (candidate_edges[i].cinfo_a->succ == NULL && candidate_edges[i].cinfo_b->pred == NULL)
		{
			candidate_edges[i].cinfo_a->succ = candidate_edges[i].cinfo_b;
			candidate_edges[i].cinfo_b->pred = candidate_edges[i].cinfo_a;
			++p;
		}
	}

	// cout << "DEBUG_ASSEMBLE_SPANS_5:" << p << endl;

	for (i = 0; i < cinfo_list.size(); ++i)
	{
		// cout << "DEBUG_ASSEMBLE_SPANS_6:" << cinfo_list[i].pred << "/" << cinfo_list[i].succ << endl;
		if (cinfo_list[i].pred != NULL)
		{
			continue;
		}
		ContourInfo *ci = &cinfo_list[i];
		std::vector<ContourInfo> cur_span;
		double width = 0;
		while (ci)
		{
			cur_span.push_back(*ci);
			width += ci->local_xrng[1] - ci->local_xrng[0];
			ci = ci->succ;
		}
		if (width > SPAN_MIN_WIDTH)
		{
			spans->push_back(cur_span);
		}
	}

	if (DEBUG_LEVEL > 0)
		visualize_spans(small, page_mask, *spans);
}


int Dewarp::round_nearest_multiple(int i, int factor)
{
	int rem = i % factor;
	if (!rem)
		return i;
	else
		return i + factor - rem;
}

void sample_spans(
	cv::Size shape,
	std::vector<std::vector<ContourInfo>> spans,
	std::vector<std::vector<cv::Point2d>> *span_points)
{
	unsigned int i, j;
	for (i = 0; i < spans.size(); ++i)
	{
		std::vector<cv::Point2d> contour_points;
		for (j = 0; j < spans[i].size(); ++j)
		{
			cv::Mat mask = cv::Mat(spans[i][j].mask);
			cv::Rect rect = cv::Rect(spans[i][j].rect);
			int height = mask.size().height;
			int width = mask.size().width;
			std::vector<int> yvals(height);
			// cout << "DEBUG_SAMPLE_SPANS_2:" << endl;
			for (int k = 0; k < height; ++k)
			{
				yvals[k] = k;
				// cout << yvals[k] << "/";
			}
			// cout << endl;

			std::vector<int> totals(width, 0);

			// cout << "DEBUG_SAMPLE_SPANS_3:" << totals.size() << endl;

			// cout << "DEBUG_SAMPLE_SPANS_4:" << endl;
			for (int c = 0; c < width; ++c)
			{
				for (int r = 0; r < height; ++r)
				{
					totals[c] += (int)mask.at<uchar>(r, c) * yvals[r];
				}
				// cout << totals[c] << "/";
			}
			// cout << endl;

			std::vector<int> mask_sum(width, 0);
			for (unsigned int k = 0; k < mask_sum.size(); ++k)
			{
				for (int l = 0; l < mask.size().height; ++l)
				{
					mask_sum[k] += (int)mask.at<uchar>(l, k);
				}
			}

			std::vector<double> means(width);
			// cout << "DEBUG_SAMPLE_SPANS_5:" << means.size() << endl;
			// cout << "DEBUG_SAMPLE_SPANS_6:" << endl;
			for (unsigned int k = 0; k < mask_sum.size(); ++k)
			{
				means[k] = (double)totals[k] / (double)mask_sum[k];
				// cout << means[k] << "/";
			}
			// cout << endl;

			int min_x = rect.x;
			int min_y = rect.y;

			// cout << "DEBUG_SAMPLE_SPANS_7:" << min_x << "/" << min_y << endl;

			int start = ((width - 1) % SPAN_PX_PER_STEP) / 2;

			// cout << "DEBUG_SAMPLE_SPANS_8:" << start << endl;

			for (int x = start; x < width; x += SPAN_PX_PER_STEP)
			{
				contour_points.push_back(cv::Point2d((double)x + (double)min_x, means[x] + (double)min_y));
			}
			// cout << "DEBUG_SAMPLE_SPANS_9:" << contour_points.size() << endl;
			// cout << "DEBUG_SAMPLE_SPANS_10:" << endl;
			// for (int k = 0; k < contour_points.size(); ++k) {
			// 	cout << contour_points[k].x << "/" << contour_points[k].y << endl;
			// }
		}

		// cout << "DEBUG_SAMPLE_SPANS_PIX2NORM_-1:" << contour_points.size() << endl;
		// cout << "DEBUG_SAMPLE_SPANS_PIX2NORM_0:" << endl;
		// for (int j = 0; j < contour_points.size(); ++j) {
		// 	cout << contour_points[j].x << "/" << contour_points[j].y << endl;
		// }

		contour_points = pix2norm(shape, contour_points);

		// cout << "DEBUG_SAMPLE_SPANS_PIX2NORM_1:" << contour_points.size() << endl;
		// cout << "DEBUG_SAMPLE_SPANS_PIX2NORM_2:" << endl;
		// for (int j = 0; j < contour_points.size(); ++j) {
		// 	cout << contour_points[j].x << "/" << contour_points[j].y << endl;
		// }
		//contour_points = norm2pix(shape, contour_points, false);
		// for (int j = 0; j < contour_points.size(); ++j) {
		// 	cout << contour_points[j].x << "/" << contour_points[j].y << endl;
		// }
		//exit(1);
		// span_points.resize(span_points.size() + 1);
		// span_points[span_points.size() - 1].insert(
		// 	span_points[span_points.size() - 1].end(),
		// 	contour_points.begin(),
		// 	contour_points.end()
		// );
		span_points->push_back(contour_points);
	}
	//cout << cv::Mat(span_points);
}

void keypoints_from_samples(
	cv::Mat small,
	cv::Mat page_mask,
	std::vector<cv::Point> page_outline,
	std::vector<std::vector<cv::Point2d>> span_points,
	std::vector<cv::Point2d> *corners,
	std::vector<std::vector<double>> *xcoords,
	std::vector<double> *ycoords)
{
	cv::Point2d all_evecs(0, 0);
	double all_weights = 0;

	for (unsigned int i = 0; i < span_points.size(); ++i)
	{
		// sai tai thang Tung loz
		cv::Mat data_pts(span_points[i].size(), 2, CV_64FC1);

		for (int j = 0; j < data_pts.size().height; ++j)
		{
			data_pts.at<double>(j, 0) = span_points[i][j].x;
			data_pts.at<double>(j, 1) = span_points[i][j].y;
		}

		// Perform PCA
		cv::PCA pca(data_pts, cv::Mat(), 0);

		cv::Point2d _evec(
			pca.eigenvectors.at<double>(0, 0),
			pca.eigenvectors.at<double>(0, 1));

		// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_1:" << _evec.x << "/" << _evec.y << endl;
		//cout << "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE" << span_points[i].size() << endl;
		double weight = sqrt(
			pow(span_points[i][0].x - span_points[i].back().x, 2) +
			pow(span_points[i][0].y - span_points[i].back().y, 2));
		//cout << "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE" << span_points[i].size() << endl;
		// type of _evec can either be Point or vector of 1 Point
		all_evecs.x += _evec.x * weight;
		all_evecs.y += _evec.y * weight;

		all_weights += weight;
	}

	cv::Point2d evec = all_evecs / all_weights;

	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_2:" << evec.x << "/" << evec.y << endl;

	cv::Point2d x_dir(evec);

	if (x_dir.x < 0)
	{
		x_dir = -x_dir;
	}
	std::vector<cv::Point2d> x_dir_vec;

	cv::Point2d y_dir(-x_dir.y, x_dir.x);

	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_3:" << x_dir.x << "/" << x_dir.y << endl;
	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_4:" << y_dir.x << "/" << y_dir.y << endl;

	std::vector<cv::Point> _pagecoords;

	cv::convexHull(page_outline, _pagecoords);
	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_5:" << _pagecoords.size() << endl;
	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_6:" << endl;
	// for (int i = 0; i < _pagecoords.size(); ++i) {
	// 	cout << _pagecoords[i].x << "/" << pagecoords[i].y << endl;
	// }

	std::vector<cv::Point2d> pagecoords(_pagecoords.size());
	for (unsigned int i = 0; i < pagecoords.size(); ++i)
	{
		pagecoords[i].x = (double)_pagecoords[i].x;
		pagecoords[i].y = (double)_pagecoords[i].y;
	}

	pagecoords = pix2norm(page_mask.size(), pagecoords);

	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_7:" << pagecoords.size() << endl;
	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_8:" << endl;
	// for ( unsigned int i = 0; i < pagecoords.size(); ++i)
	// {
	// 	cout << pagecoords[i].x << "/" << pagecoords[i].y << endl;
	// }

	// cv::Mat px_coords = cv::Mat(pagecoords) * cv::Mat(x_dir);
	// cv::Mat py_coords = cv::Mat(pagecoords) * cv::Mat(y_dir);

	std::vector<double> px_coords(pagecoords.size());
	std::vector<double> py_coords(pagecoords.size());

	for (unsigned int i = 0; i < pagecoords.size(); ++i)
	{
		px_coords[i] = pagecoords[i].x * x_dir.x + pagecoords[i].y * x_dir.y;
		py_coords[i] = pagecoords[i].x * y_dir.x + pagecoords[i].y * y_dir.y;
	}

	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_11" << endl;
	// for (int i = 0; i < pagecoords.size(); ++i) {
	// 	cout << px_coords[i] << "/";
	// }
	// cout << endl;
	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_12" << endl;
	// for (int i = 0; i < pagecoords.size(); ++i) {
	// 	cout << py_coords[i] << "/";
	// }
	// cout << endl;

	double px0, px1, py0, py1;
	// std::max_element(px_coords.begin(), px_coords.end());
	// std::max_element(px_coords.begin(), px_coords.end());

	cv::minMaxLoc(px_coords, &px0, &px1);
	cv::minMaxLoc(py_coords, &py0, &py1);

	// cout << "DEBUG_DEBUG_KEYPOINTS_FROM_SAMPLE_14" << px0 << endl;
	// cout << "DEBUG_DEBUG_KEYPOINTS_FROM_SAMPLE_15" << px1 << endl;
	// cout << "DEBUG_DEBUG_KEYPOINTS_FROM_SAMPLE_16" << py0 << endl;
	// cout << "DEBUG_DEBUG_KEYPOINTS_FROM_SAMPLE_17" << py1 << endl;

	cv::Point2d p00 = px0 * x_dir + py0 * y_dir;
	cv::Point2d p10 = px1 * x_dir + py0 * y_dir;
	cv::Point2d p11 = px1 * x_dir + py1 * y_dir;
	cv::Point2d p01 = px0 * x_dir + py1 * y_dir;

	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_18:" << p00.x << "/" << p00.y << endl;
	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_19:" << p10.x << "/" << p10.y << endl;
	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_20:" << p11.x << "/" << p11.y << endl;
	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_21:" << p01.x << "/" << p01.y << endl;

	corners->push_back(p00);
	corners->push_back(p10);
	corners->push_back(p11);
	corners->push_back(p01);

	// cout << "DEBUG_KEYPOINTS_FROM_SAMPLE_22" << endl;
	// for (int i = 0; i < corners.size(); ++i) {
	// 	cout << corners[i].x << "/" << corners[i].y << endl;
	// }

	for (unsigned int i = 0; i < span_points.size(); ++i)
	{
		std::vector<double> _px_coords(span_points[i].size());
		std::vector<double> _py_coords(span_points[i].size());
		for (unsigned int j = 0; j < span_points[i].size(); ++j)
		{
			_px_coords[j] = span_points[i][j].x * x_dir.x + span_points[i][j].y * x_dir.y;
			_py_coords[j] = span_points[i][j].x * y_dir.x + span_points[i][j].y * y_dir.y;
		}
		double _py_coords_mean = 0;
		for (unsigned int k = 0; k < _py_coords.size(); ++k)
		{
			_py_coords_mean += _py_coords[k];
			_px_coords[k] -= px0;
		}
		_py_coords_mean /= _py_coords.size();
		xcoords->push_back(_px_coords);
		ycoords->push_back(_py_coords_mean - py0);
	}
	if (DEBUG_LEVEL > 0)
		visualize_span_points(small, span_points, *corners);
}

void get_default_params(
	std::vector<cv::Point2d> corners,
	std::vector<double> ycoords,
	std::vector<std::vector<double>> xcoords,
	double *rough_dims,
	std::vector<int> *span_counts,
	std::vector<double> *params)
{
	double page_width = sqrt(
		pow(corners[1].x - corners[0].x, 2) +
		pow(corners[1].y - corners[0].y, 2));
	double page_height = sqrt(
		pow(corners[corners.size() - 1].x - corners[0].x, 2) +
		pow(corners[corners.size() - 1].y - corners[0].y, 2));
	//printf("page_width = %lf\n", page_width);
	//printf("page_height = %lf\n", page_height);

	rough_dims[0] = page_width;
	rough_dims[1] = page_height;

	std::vector<cv::Point3d> corners_object3d;
	corners_object3d.push_back(cv::Point3d(0, 0, 0));
	corners_object3d.push_back(cv::Point3d(page_width, 0, 0));
	corners_object3d.push_back(cv::Point3d(page_width, page_height, 0));
	corners_object3d.push_back(cv::Point3d(0, page_height, 0));

	// cv::Mat cameraMatrix = cv::Mat::zeros(3,3,cv::DataType<double>::type);
	// cameraMatrix.at<double>(0, 0) = FOCAL_LENGTH;
	// cameraMatrix.at<double>(1, 1) = FOCAL_LENGTH;
	// cameraMatrix.at<double>(2, 2) = 1;

	cv::Mat rvec; //(3, 1, cv::DataType<double>::type);
	cv::Mat tvec; //(3, 1, cv::DataType<double>::type);
	// std::vector<int> rvec = std::vector<int>(RVEC_IDX(pvec));
	// std::vector<int> tvec = std::vector<int>(TVEC_IDX(pvec));
	cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type);
	cameraMatrix = 0;
	cameraMatrix.at<double>(0, 0) = FOCAL_LENGTH;
	cameraMatrix.at<double>(1, 1) = FOCAL_LENGTH;
	cameraMatrix.at<double>(2, 2) = 1;
	cv::solvePnP(corners_object3d, corners, cameraMatrix, cv::Mat::zeros(5, 1, CV_64F), rvec, tvec);
	//printf("rvec.cols = %d, rvec.rows = %d\n", rvec.cols, rvec.rows);
	//printf("tvec.cols = %d, tvec.rows = %d\n", tvec.cols, tvec.rows);

	params->push_back(rvec.at<double>(0, 0));
	params->push_back(rvec.at<double>(1, 0));
	params->push_back(rvec.at<double>(2, 0));
	params->push_back(tvec.at<double>(0, 0));
	params->push_back(tvec.at<double>(1, 0));
	params->push_back(tvec.at<double>(2, 0));
	std::vector<double> news(2, 0);
	params->insert(params->end(), news.begin(), news.end());
	params->insert(params->end(), ycoords.begin(), ycoords.end()); // buggy

	for (unsigned int i = 0; i < xcoords.size(); ++i)
	{
		span_counts->push_back(xcoords[i].size());
		params->insert(params->end(), xcoords[i].begin(), xcoords[i].end());
	}
}

void Optimize::make_keypoint_index(std::vector<int> span_counts)
{
	int nspans = (int)span_counts.size();

	int npts, i, j;
	npts = 0;

	int start, end, count;
	for (i = 0; i < (int)span_counts.size(); i++)
		npts += span_counts[i];
	//printf("SPAN COUNT = %d\n", npts);
	//exit(1);
	keypoint_index[0] = std::vector<int>(npts + 1, 0);
	keypoint_index[1] = std::vector<int>(npts + 1, 0);
	//keypoint = std::vector<cv::Point2d>(npts + 1);
	start = 1;
	////printf("keypoint index = %lu\n", keypoint_index[0].size());

	for (i = 0; i < (int)span_counts.size(); i++)
	{
		count = span_counts[i];
		end = start + count;
		////printf("start + end = ")
		for (j = start; j < end; j++)
			keypoint_index[1][j] = 8 + i;
		start = end;
	}

	//cout << "MAKE_KEYPOINT_INDEX_1" << endl;

	for (i = 1; i < npts; i++)
		keypoint_index[0][i] = i - 1 + 8 + nspans;
}

void polyval(std::vector<double> poly, std::vector<cv::Point2d> xy_coords, std::vector<cv::Point3d> *objpoints)
{
	double x2, x3, z;
	for (unsigned int i = 0; i < xy_coords.size(); i++)
	{
		//printf
		x2 = xy_coords[i].x * xy_coords[i].x;
		x3 = x2 * xy_coords[i].x;
		z = x3 * poly[0] + x2 * poly[1] + xy_coords[i].x * poly[2] + poly[3];
		objpoints->push_back(cv::Point3d((double)xy_coords[i].x, (double)xy_coords[i].y, (double)z));
	}
}


void project_xy(std::vector<cv::Point2d> &xy_coords, std::vector<double> pvec, std::vector<cv::Point2d> *imagepoints)
{
	// alpha, beta = tuple(pvec[CUBIC_IDX])

	double alpha, beta;

	std::vector<double> rvec = std::vector<double>(pvec.begin(), pvec.begin() + 3);
	std::vector<double> tvec = std::vector<double>(pvec.begin() + 3, pvec.begin() + 6);

	cv::Mat K(3, 3, cv::DataType<double>::type);
	K = 0;
	K.at<double>(0, 0) = FOCAL_LENGTH;
	K.at<double>(1, 1) = FOCAL_LENGTH;
	K.at<double>(2, 2) = 1;
	cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);
	distCoeffs.at<double>(0) = 0;
	distCoeffs.at<double>(1) = 0;
	distCoeffs.at<double>(2) = 0;
	distCoeffs.at<double>(3) = 0;
	distCoeffs.at<double>(4) = 0;
	alpha = pvec[6];
	beta = pvec[7];
	std::vector<double> poly;
	std::vector<cv::Point3d> objpoints; //(xy_coords.size(), 0);
	poly.push_back(alpha + beta);
	poly.push_back(-2 * alpha - beta);
	poly.push_back(alpha);
	poly.push_back(0);
	polyval(poly, xy_coords, &objpoints);

	cv::projectPoints(objpoints, rvec, tvec, K, distCoeffs, *imagepoints);
}

void project_keypoints(const std::vector<double> &pvec, std::vector<int> *keypoint_index, std::vector<cv::Point2d> *imagepoints)
{

	std::vector<cv::Point2d> xy_coords(keypoint_index[0].size());

	unsigned int i;
	xy_coords[0].x = 0;
	xy_coords[0].y = 0;

	for (i = 1; i < keypoint_index[0].size(); i++)
	{
		xy_coords[i].x = pvec[keypoint_index[0][i]];
		xy_coords[i].y = pvec[keypoint_index[1][i]];
	}

	project_xy(xy_coords, pvec, imagepoints);
}



int loop = 0;
double min_loop;

double be_like_target(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{

	std::vector<cv::Point2d> imagepoints;
	double min = 0;
	project_keypoints(x, keypoint_index, &imagepoints);
	loop = loop + 1;
	for (unsigned int i = 0; i < imagepoints.size() - 1; i++)
	{
		min += pow(cv::norm(span_points_flat[i] - imagepoints[i]), 2);
	}
 //   __android_log_print(ANDROID_LOG_ERROR, "DEBUG_get_page_extent", "%lf", min);
	if (DEBUG_LEVEL)
		printf("loop = %d, min = %lf\n", loop, min);
	//min_loop = min;
	return min;
};

double Optimize::Minimize(std::vector<double> params)
{
	// std::vector<double> dstpoint(400);
	// for (unsigned int i = 0; i < dstpoint.size(); i++)
	//     dstpoint[i] = i;
	nlopt::opt opt(nlopt::LN_PRAXIS, params.size());
	std::vector<double> lb(params.size(), -5.0);
	std::vector<double> lu(params.size(), 5.0);
	opt.set_lower_bounds(lb);
	opt.set_upper_bounds(lu);
	opt.set_min_objective(be_like_target, NULL);
	opt.set_xtol_rel(0.02);
	//opt.set_ftol_rel(1e-5);
	std::vector<cv::Point2d> imagepoints;
	opt.set_maxtime(1000);
	std::vector<double> x;
	x = params;
	double opt_f = 100;
	opt.optimize(x, opt_f);
	for (unsigned int i = 0; i < x.size(); i++)
	{
		//printf("out_page_dims[%u] = %lf\n", i, x[i]);
		out_params.push_back(x[i]);
	}
	return 0;
}

double be_like_target1(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
	std::vector<cv::Point2d> xy_coords;
	xy_coords.push_back(cv::Point(x[0], x[1]));
	std::vector<cv::Point2d> imagepoints;
	double min = 0;
	project_xy(xy_coords, out_params, &imagepoints);
	for (unsigned int i = 0; i < imagepoints.size(); i++)
	{
		min += pow(cv::norm(corners[2] - imagepoints[i]), 2);
	}
	return min;
}
double Optimize::get_page_dims(std::vector<double> params)
{

	nlopt::opt opt(nlopt::LN_PRAXIS, 2);
	std::vector<double> lb(2, -1280.0);
	std::vector<double> lu(2, 1280.0);
	opt.set_lower_bounds(lb);
	opt.set_upper_bounds(lu);
	opt.set_min_objective(be_like_target1, NULL);
	opt.set_xtol_rel(1e-3);
	std::vector<double> x;
	for (unsigned int i = 0; i < 2; i++)
	{
		x.push_back(dims[i]);
	}
	opt.set_maxtime(100);
	double opt_f;
	opt.optimize(x, opt_f);
	for (unsigned int i = 0; i < x.size(); i++)
	{
		out_page_dims.push_back(x[i]);
	}
	return 0;
}

int round_nearest_multiple(double i, int factor)
{
	int _i = (int)i;
	int rem = _i % factor;
	if (!rem)
	{
		return _i;
	}
	else
	{
		return _i + factor - rem;
	}
}

void Optimize::remap_image(string name, cv::Mat img, cv::Mat small, cv::Mat &thresh, std::vector<double> page_dims, std::vector<double> params/*, string outfile_prefix*/)
{

	double height = page_dims[1] / 2 * OUTPUT_ZOOM * img.size().height;

	int _height = round_nearest_multiple(height, REMAP_DECIMATE);
	int _width = round_nearest_multiple((double)_height * page_dims[0] / page_dims[1], REMAP_DECIMATE);

	int height_small = _height / REMAP_DECIMATE;
	int width_small = _width / REMAP_DECIMATE;

	int currentIndex;
	std::vector<double> page_x_range(width_small, 0);
	std::vector<double> page_y_range(height_small, 0);
	double x_range_space = page_dims[0] / (width_small - 1);
	double y_range_space = page_dims[1] / (height_small - 1);
	for (int i = 1; i < width_small; i++)
	{
		page_x_range[i] = page_x_range[i - 1] + x_range_space;
	}
	for (int j = 1; j < height_small; j++)
	{
		page_y_range[j] = page_y_range[j - 1] + y_range_space;
	}

	std::vector<cv::Point2d> page_xy_coords(width_small * height_small);
	// x_coord
	for (int j = 0; j < height_small; j++)
	{
		// y_coord
		for (int i = 0; i < width_small; i++)
		{
			currentIndex = j * width_small + i;
			page_xy_coords[currentIndex].x = page_x_range[i];
			page_xy_coords[currentIndex].y = page_y_range[j];
		}
	}

	std::vector<cv::Point2d> image_points;
	project_xy(page_xy_coords, params, &image_points);

	image_points = norm2pix(img.size(), image_points, false);

	cv::Mat image_x_coords(cv::Size(width_small, height_small), CV_32FC1);
	cv::Mat image_y_coords(cv::Size(width_small, height_small), CV_32FC1);
	for (int j = 0; j < image_x_coords.rows; j++)
	{
		for (int i = 0; i < image_x_coords.cols; i++)
		{
			currentIndex = j * width_small + i;
			image_x_coords.at<float>(j, i) = (float)image_points[currentIndex].x;
			image_y_coords.at<float>(j, i) = (float)image_points[currentIndex].y;
		}
	}

	cv::Mat image_x_out, image_y_out;

	cv::resize(image_x_coords, image_x_out, cv::Size(_width, _height), 0, 0, cv::INTER_CUBIC);
	cv::resize(image_y_coords, image_y_out, cv::Size(_width, _height), 0, 0, cv::INTER_CUBIC);
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_RGB2GRAY);
	cv::Mat remapped;
	cv::remap(img_gray, remapped, image_x_out, image_y_out, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
	if (DEBUG_LEVEL)
		cv::imwrite("result_remap.png", remapped);
//	cv::adaptiveThreshold(remapped, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, ADAPTIVE_WINSZ, 25);
	thresh = remapped.clone();
}

int page_dewarp(cv::Mat img_src, cv::Mat &img_dst, std::vector <cv::Point2f> line_point/*, string outfile_prefix*/)
{
	cv::Mat small, page_mask;
	std::vector<cv::Point> page_outline;
	std::vector<ContourInfo> contours_out;
	std::vector<std::vector<ContourInfo>> spans, spans2;
	std::vector<std::vector<cv::Point2d>> span_points;
	string filename("abc");
	resize_to_screen(img_src, &small);

	get_page_extents(small, line_point, page_mask, &page_outline);

	get_contours_s(filename, small, page_mask, 0, contours_out);

	assemble_spans(filename, small, page_mask, contours_out, &spans);

	if (spans.size() < 3)
	{
		//printf("detecting lines because only %lu text spans\n", spans.size());
		get_contours_s(filename, small, page_mask, 1, contours_out);
		assemble_spans(filename, small, page_mask, contours_out, &spans);
		if (spans2.size() > spans.size())
			spans = spans2;
	}

	if (spans.size() < 1)
	{
		//printf("skipping because only %lu continue\n",  spans.size());
		return 0;
	}

	sample_spans(small.size(), spans, &span_points);

	std::vector<std::vector<double>> xcoords;
	std::vector<double> ycoords;

	keypoints_from_samples(small, page_mask, page_outline, span_points, &corners, &xcoords, &ycoords);
	double rough_dims[2];
	std::vector<int> span_counts;
	get_default_params(corners, ycoords, xcoords, rough_dims, &span_counts, &params);
	span_points_flat.push_back(corners[0]);
	for (unsigned int i = 0; i < span_points.size(); ++i)
	{
		span_points_flat.insert(span_points_flat.end(), span_points[i].begin(), span_points[i].end());
	}
	Optimize Object;
	Object.span_counts = span_counts;
	Object.make_keypoint_index(span_counts);

	dims[0] = rough_dims[0];
	dims[1] = rough_dims[1];
	std::vector<cv::Point2d> imagepoints_debug;

	project_keypoints(params, keypoint_index, &imagepoints_debug);
	if (DEBUG_LEVEL)
		draw_correspondences("before_optimize.png", small, span_points_flat, imagepoints_debug);
	Object.Minimize(params);
	imagepoints_debug.clear();
	if (DEBUG_LEVEL) {
		project_keypoints(out_params, keypoint_index, &imagepoints_debug);
		draw_correspondences("after.png", small, span_points_flat, imagepoints_debug);
	}
	Object.get_page_dims(out_params);
	std::vector<double> dim_vector{out_page_dims[0], out_page_dims[1]};
	cout << "DEBUG_BEFORE_REMAP_IMAGE" << endl;
	Object.remap_image("output", img_src, small, img_dst, dim_vector, out_params/*, outfile_prefix*/);
	cout << "DEBUG_AFTER_REMAP_IMAGE" << endl;
	span_points_flat.clear();
	corners.clear();
	keypoint_index[0].clear();
	keypoint_index[1].clear();
	keypoint.clear();
	params.clear();
	out_params.clear();
	out_page_dims.clear();
	return 0;
}

extern "C" {

JNIEXPORT jlong JNICALL Java_com_example_builddewarp_CaptureImage_00024ImageSave_dewarpImage
		(JNIEnv*, jobject, jlong src, jlong dst){
	Mat& img_src = *(Mat*) src;
	Mat& img_dst = *(Mat*) dst;
	ofstream outfile;
	Mat *mat;
	std::vector<cv::Point2f> point;
	page_dewarp(img_src, img_dst, line_point);
	line_point.clear();
	mat = &img_dst;
/*	__android_log_print(ANDROID_LOG_ERROR, "DEBUG_get_page_extent1", "%p", mat);
	__android_log_print(ANDROID_LOG_ERROR, "DEBUG_get_page_extent2", "%d", mat->cols);
	__android_log_print(ANDROID_LOG_ERROR, "DEBUG_get_page_extent3", "%d", mat->rows);*/
	return (jlong) mat;
};
}
