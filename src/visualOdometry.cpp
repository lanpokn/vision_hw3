#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "slamBase.h"

// 给定index，读取一帧数据
FRAME readFrame( int index, ParameterReader& pd );
// 度量运动的大小
double normofTransform( cv::Mat rvec, cv::Mat tvec );

int main( int argc, char** argv )
{
    ParameterReader pd;
    int startIndex  =   atoi( pd.getData( "start_index" ).c_str() );
    int endIndex    =   atoi( pd.getData( "end_index"   ).c_str() );

    // initialize
    cout<<"Initializing ..."<<endl;
    int currIndex = startIndex; // 当前索引为currIndex
    FRAME lastFrame = readFrame( currIndex, pd ); // 上一帧数据
    // 我们总是在比较currFrame和lastFrame
    string detector = pd.getData( "detector" );
    string descriptor = pd.getData( "descriptor" );
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    computeKeyPointsAndDesp( lastFrame, detector, descriptor );
    ///
    PointCloud::Ptr cloud = image2PointCloud( lastFrame.rgb, lastFrame.depth, camera );

    pcl::visualization::CloudViewer viewer("viewer");

    // 是否显示点云
    bool visualize = pd.getData("visualize_pointcloud")==string("yes");

    int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    double max_norm = atof( pd.getData("max_norm").c_str() );

    ofstream outfile("../kitti03/result/my.txt", ios::out|ios::trunc);
    for ( currIndex=startIndex+1; currIndex<endIndex; currIndex++ )
    {
        cout<<"Reading files "<<currIndex<<endl;
        FRAME currFrame = readFrame( currIndex,pd ); // 读取currFrame
        computeKeyPointsAndDesp( currFrame, detector, descriptor );
        // 比较currFrame 和 lastFrame
        RESULT_OF_PNP result = estimateMotion( lastFrame, currFrame, camera );
        cout<<"result.inliers:"<<result.inliers<<endl;
        cout<<"r"<<result.rvec<<endl<<"t"<<result.tvec<<endl;
        if ( result.inliers < min_inliers ) //inliers不够，放弃该帧
            continue;
        // 计算运动范围是否太大
        double norm = normofTransform(result.rvec, result.tvec);
        cout<<"norm = "<<norm<<endl;
        if ( norm >= max_norm )
            continue;
        Eigen::Isometry3d T = cvMat2Eigen( result.rvec, result.tvec );
        // cout<<"T="<<T.matrix()<<endl;
        outfile<<T(0,0)<<""<<T(0,1)<<""<<T(0,2)<<""<<T(0,3)<<endl;
        outfile<<T(1,0)<<""<<T(1,1)<<""<<T(1,2)<<""<<T(1,3)<<endl;
        outfile<<T(2,0)<<""<<T(2,1)<<""<<T(2,2)<<""<<T(2,3)<<endl;
        // cloud = joinPointCloud( cloud, currFrame, T.inverse(), camera );
        cloud = joinPointCloud( cloud, currFrame, T, camera );

        if ( visualize == true )
            viewer.showCloud( cloud );

        lastFrame = currFrame;
        pcl::io::savePCDFile( "../kitti03/data/result.pcd", *cloud );
    }
    outfile.close();
    
    pcl::io::savePCDFile( "../kitti03/data/result.pcd", *cloud );
    return 0;
}

FRAME readFrame( int index, ParameterReader& pd )
{
    FRAME f;
    string rgbDir   =   pd.getData("rgb_dir");
    string depthDir =   pd.getData("depth_dir");

    string rgbExt   =   pd.getData("rgb_extension");
    string depthExt =   pd.getData("depth_extension");
    string numzero;
    if ( index >= 0 && index <= 9 )
    {
        numzero = "00000";
    }
    else if ( index >= 10 && index <= 99 )
    {
        numzero = "0000";
    }
    else if ( index >= 100 && index <= 999 )
    {
        numzero = "000";
    }
    else if ( index >= 1000 && index <= 9999 )
    {
        numzero = "00";
    }
    else if ( index >= 10000 && index <= 99999 )
    {
        numzero = "0";
    }
    stringstream ss;
    ss<<rgbDir<<numzero<<index<<rgbExt;
    string filename;
    ss>>filename;
    f.rgb = cv::imread( filename );

    ss.clear();
    filename.clear();
    ss<<depthDir<<numzero<<index<<depthExt;
    ss>>filename;

    f.depth = cv::imread( filename, -1 );
    return f;
}

double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}