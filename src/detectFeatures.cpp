/*************************************************************************
  2     > File Name: detectFeatures.cpp
  3     > Author: xiang gao
  4     > Mail: gaoxiang12@mails.tsinghua.edu.cn
  5     > 特征提取与匹配
  6     > Created Time: 2015年07月18日 星期六 16时00分21秒
  7  ************************************************************************/
   
#include<iostream>
#include "slamBase.h"
using namespace std;
using namespace cv;

// OpenCV 特征检测模块
#include <opencv2/features2d/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
  
int main( int argc, char** argv )
{
    // 声明并从data文件夹里读取两个rgb与深度图
    cv::Mat rgb1 = cv::imread( "../kitti03/image_0/000000.png");
    cv::Mat rgb2 = cv::imread( "../kitti03/image_0/000001.png");
    cv::Mat depth1 = cv::imread( "../kitti03/depth/000000.png", -1);
    cv::Mat depth2 = cv::imread( "../kitti03/depth/000001.png", -1);

    // 声明特征提取器与描述子提取器
    // detector = cv::FeatureDetector::create("ORB");
    // descriptor = cv::DescriptorExtractor::create("ORB");
    cv::Ptr<FeatureDetector> detector = ORB::create();
    cv::Ptr<DescriptorExtractor> descriptor = ORB::create();
    cout<<"< Extracting keypoints from images using ORB..."<<endl;

    // 构建提取器，默认两者都为 ORB

    vector< cv::KeyPoint > kp1, kp2; //关键点
    detector->detect( rgb1, kp1 );  //提取关键点
    detector->detect( rgb2, kp2 );

    cout<<"Key points of two images: "<<kp1.size()<<", "<<kp2.size()<<endl;
    

    // 可视化， 显示关键点
    cv::Mat imgShow;
    cv::drawKeypoints( rgb1, kp1, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    cv::imshow( "keypoints", imgShow );
    cv::imwrite( "../kitti03/data/keypoints.png", imgShow );
    cv::waitKey(0); //暂停等待一个按键

    // 计算描述子
    cv::Mat desp1, desp2;
    descriptor->compute( rgb1, kp1, desp1 );
    descriptor->compute( rgb2, kp2, desp2 );

    // 匹配描述子
    vector< cv::DMatch > matches; 
    cv::BFMatcher matcher;
    matcher.match( desp1, desp2, matches );
    cout<<"Find total "<<matches.size()<<" matches."<<endl;

    // 可视化：显示匹配的特征
    cv::Mat imgMatches;
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matches, imgMatches );
    cv::imshow( "matches", imgMatches );
    cv::imwrite( "../kitti03/data/matches.png", imgMatches );
    cv::waitKey( 0 );

    // 筛选匹配，把距离太大的去掉
    // 这里使用的准则是去掉大于四倍最小距离的匹配
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }
    cout<<"min dis = "<<minDis<<endl;

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < 10*minDis)
            goodMatches.push_back( matches[i] );
    }

    // 显示 good matches
    cout<<"good matches="<<goodMatches.size()<<endl;
    cv::drawMatches( rgb1, kp1, rgb2, kp2, goodMatches, imgMatches );
    cv::imshow( "good matches", imgMatches );
    cv::imwrite( "../kitti03/data/good_matches.png", imgMatches );
    cv::waitKey(0);

    // 计算图像间的运动关系
    // 关键函数：cv::solvePnPRansac()
    // 为调用此函数准备必要的参数
    
    // 第一个帧的三维点
    vector<cv::Point3f> pts_obj;
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;

    // 相机内参
    // const double camera_factor = 10000;
    // const double camera_cx = 609.5593;
    // const double camera_cy = 172.854;
    // const double camera_fx = 721.5377;
    // const double camera_fy = 721.5377;

    CAMERA_INTRINSIC_PARAMETERS C;
    C.cx = 609.5593;
    C.cy = 172.854;
    C.fx = 721.5377;
    C.fy = 721.5377;
    C.scale = 10000;

    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p = kp1[goodMatches[i].queryIdx].pt;
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d = depth1.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        pts_img.push_back( cv::Point2f( kp2[goodMatches[i].trainIdx].pt ) );

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, C );
        pts_obj.push_back( pd );
    }

    double camera_matrix_data[3][3] = {
        {C.fx, 0, C.cx},
        {0, C.fy, C.cy},
        {0, 0, 1}
    };

    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // 求解pnp
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, (0.9899999999999999911), inliers );
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, (0.9899999999999999911), inliers );

    cout<<"inliers: "<<inliers.rows<<endl;
    cout<<"R="<<rvec<<endl;
    cout<<"t="<<tvec<<endl;

    // 画出inliers匹配 
    vector< cv::DMatch > matchesShow;
    for (size_t i=0; i<inliers.rows; i++)
    {
        matchesShow.push_back( goodMatches[inliers.ptr<int>(i)[0]] );    
    }
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matchesShow, imgMatches );
    cv::imshow( "inlier matches", imgMatches );
    cv::imwrite( "../kitti03/data/inliers.png", imgMatches );
    cv::waitKey( 0 );

    return 0;
}