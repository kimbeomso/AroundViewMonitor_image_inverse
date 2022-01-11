#include "ocamcalib_undistort/ocam_functions.h"
#include "ocamcalib_undistort/Parameters.h"
 
#include <iostream>
#include <string>
#include <exception>

#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseArray.h>

#include <cv_bridge/cv_bridge.h>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

#include <std_msgs/Float32MultiArray.h>
#include <pcl_ros/point_cloud.h>
#include <algorithm>

// #include <ocamcalib_undistort/PhantomVisionNetMsg.h> 
// #include <ocamcalib_undistort/VisionPhantomnetData.h>
// #include <ocamcalib_undistort/VisionPhantomnetDataList.h>
#include <ocamcalib_undistort/ParkingPhantomnetData.h>
#include <ocamcalib_undistort/ParkingPhantomnetDetection.h>
#include <ocamcalib_undistort/Bounding_Box.h> 

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/io/io.hpp>
#include <boost/program_options.hpp>

using namespace cv;
using namespace std;

//선택지 6개
#define DIRECTION 0         //side: 0, center: 1, both: 2
#define SEG_IMG_PUB 1       // 1: true, 2: false

#define AVM_IMG_WIDTH  400
#define AVM_IMG_HEIGHT 400
// #define CROP_TOP 12
#define PADDING_CENTER_BOTTOM 128
#define PADDING_REAR_BOTTOM 128

#define PADDING_VALUE 128
bool Padding_UP = true;  //1 up, 0 down

#define FULL_IMG_RESOL_WIDTH  1920
#define FULL_IMG_RESOL_HEIGHT 1208

#define RESIZE_BAG_TO_ORG 3.75 
#define CROP_ROI_WIDTH   1920//(int)((double)512 * RESIZE_BAG_TO_ORG)
#define CROP_ROI_HEIGHT  1080//(int)((double)288 * RESIZE_BAG_TO_ORG)

#define REAL_OCCUPANCY_SIZE_X 16.5    // AVM_IMG_WIDTH 400PIX == 25Meter
#define REAL_OCCUPANCY_SIZE_Y 16.5    // AVM_IMG_HEIGHT 400PIX == 25Meter

#define PIXEL_PER_METER  AVM_IMG_WIDTH/REAL_OCCUPANCY_SIZE_X           //400PIX / 16m                     원래작업 : 16pix/m에서 25pix/m
#define METER_PER_PIXEL  (double)(1.0/(double)(PIXEL_PER_METER)) // 0.0625                                        // 0.0625

//////////////////////////////////////////////////////
#define M_SPACE_WIDTH 2.65  // 2.6
#define M_SPACE_LENGTH 7.0  // 5.2
#define FREE_ERASE_GAIN 1.2

#define FRONT 0
#define REAR 1
#define LEFT 2
#define RIGHT 3

#define THREE_CH 0 
#define ONE_CH 1

#define PARKING_SPACE 21
#define OBJ_DET_CONFIDENCE 0.1
#define ENLARGED_BOX 15

bool m_flag_both_dir = true;  
int m_flag_dir = 2 ;           // side :0, center :1, both :2 

std::string calibration_front ="/home/beomsoo/catkin_ws/src/ocamcalib_undistort/include/calib_results_phantom_190_028_front.txt";
std::string calibration_left = "/home/beomsoo/catkin_ws/src/ocamcalib_undistort/include/calib_results_phantom_190_022_left.txt";
std::string calibration_right ="/home/beomsoo/catkin_ws/src/ocamcalib_undistort/include/calib_results_phantom_190_023_right.txt";
std::string calibration_rear = "/home/beomsoo/catkin_ws/src/ocamcalib_undistort/include/calib_results_phantom_190_029_rear.txt" ;      

ros::Subscriber sub_tmp;
ros::Subscriber Sub_phantom_side_left, Sub_phantom_side_right, Sub_phantom_front_center, Sub_phantom_rear_center;
ros::Subscriber Sub_phantom_left_seg, Sub_phantom_right_seg, Sub_phantom_front_seg, Sub_phantom_rear_seg;
ros::Subscriber Sub_phantom_DR_Path, Sub_parkingGoal;
ros::Subscriber Sub_phantom_tmp;

ros::Publisher Pub_AVM_side_img, Pub_AVM_center_img,Pub_AVM_side_seg_img, Pub_AVM_center_seg_img, Pub_AVM_seg_img_gray, Pub_AVM_DR, Pub_Bounding_Box;
ros::Publisher Pub_AVM_seg_img, Pub_Boundingbox;

cv::Mat AVM_left = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);// = cv::Mat::zeros(450,450, CV_8UC3);
cv::Mat AVM_right = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);// = cv::Mat::zeros(450,450, CV_8UC3);
cv::Mat AVM_front = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);
cv::Mat AVM_rear = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);

cv::Mat AVM_seg_front = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);
cv::Mat AVM_seg_rear = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);
cv::Mat AVM_seg_left = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);
cv::Mat AVM_seg_right = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);
cv::Mat AVM_seg_left_gray = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC1);
cv::Mat AVM_seg_right_gray = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC1);

cv::Mat aggregated_side_img    = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT , CV_8UC3);
cv::Mat aggregated_center_img    = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT , CV_8UC3);

cv::Mat aggregated_side_seg_img = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT , CV_8UC3);
cv::Mat aggregated_center_seg_img = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT , CV_8UC3);

cv::Mat aggregated_seg_img_gray = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT , CV_8UC1);
cv::Mat aggregated_img= cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT , CV_8UC3);

cv::Mat temp1 = cv::Mat::zeros( 288,512 , CV_8UC3);
cv::Mat temp2 = cv::Mat::zeros( 288,512 , CV_8UC3);
cv::Mat temp3 = cv::Mat::zeros( 288,512 , CV_8UC3);
cv::Mat temp4 = cv::Mat::zeros( 288,512 , CV_8UC3);

// Calibration Results
ocam_model front_model;
ocam_model  left_model;
ocam_model right_model;
ocam_model  rear_model;

#define M_DEG2RAD  3.1415926 / 180.0

// std::vector<DETECTION *> Front_detection;
// std::vector<DETECTION *> Rear_detection;
// std::vector<ParkingPhantomnetDetection> Left_detection;
// std::vector<DETECTION *> Right_detection;

// ParkingPhantomnetDetection Left_detection;
typedef struct DETECTIONS{
    char classification;
    float probability;
    float x;
    float y;
    float width;
    float height;
};
std::vector<DETECTIONS> Left_detection;
std::vector<DETECTIONS> Rear_detection;
std::vector<DETECTIONS> Right_detection;
std::vector<DETECTIONS> Front_detection;

// // Extrinsic Parameters
// double M_front_param[6] = {0.688 * M_DEG2RAD,  21.631 * M_DEG2RAD,   3.103* M_DEG2RAD   ,1.905,   0.033, 0.707 };
// double M_left_param[6] =  {1.133 * M_DEG2RAD,  19.535 * M_DEG2RAD,   92.160* M_DEG2RAD  ,0.0,     1.034, 0.974 };
// double M_right_param[6] = {3.440 * M_DEG2RAD,  18.273 * M_DEG2RAD,  -86.127* M_DEG2RAD  ,0.0,    -1.034, 0.988 };
// double M_back_param[6] =  {0.752 * M_DEG2RAD,  31.238 * M_DEG2RAD,  -178.189* M_DEG2RAD ,-2.973, -0.065, 0.883 };

// // New Extrinsic Parameters 15 mm --> more corect than 9 mm
double M_front_param[6] = {0.672 * M_DEG2RAD,  21.378 * M_DEG2RAD,   1.462* M_DEG2RAD   ,   1.885,   0.038, 0.686 };
double M_left_param[6] =  {0.963 * M_DEG2RAD,  19.283 * M_DEG2RAD,   91.702* M_DEG2RAD  ,   0.0,    1.059, 0.978 };
double M_right_param[6] = {1.714 * M_DEG2RAD,  19.713 * M_DEG2RAD,  -87.631* M_DEG2RAD  ,   0.0,    -1.059, 0.972 };
double M_back_param[6] =  {-0.257 * M_DEG2RAD, 32.645 * M_DEG2RAD,  179.773* M_DEG2RAD ,   -3.002, -0.033, 0.922 };

// // New Extrinsic Parameters 9 mm
// double M_front_param[6] = {0.617 * M_DEG2RAD,  21.397 * M_DEG2RAD,   1.381* M_DEG2RAD   ,   1.880,   0.038, 0.689 };
// double M_left_param[6] =  {0.970 * M_DEG2RAD,  19.231 * M_DEG2RAD,   91.699* M_DEG2RAD  ,   0.0,    1.053, 0.979 };
// double M_right_param[6] = {1.659 * M_DEG2RAD,  19.690 * M_DEG2RAD,  -87.631* M_DEG2RAD  ,   0.0,    -1.053, 0.979 };
// double M_back_param[6] =  {-0.150 * M_DEG2RAD, 32.634 * M_DEG2RAD,  179.708* M_DEG2RAD ,   -2.997, -0.033, 0.924 };

int flag = 0; 
int resolution = 1;

// occupancy grid map for path planning
nav_msgs::OccupancyGrid occupancyGridMap;
ros::Publisher Pub_occupancyGridMap;
int m_dimension = 35;
double m_gridResol = 0.2;//0.25;
const int m_gridDim = (int)(m_dimension*(int)(1/m_gridResol));
// int num_obsL[140][140] = {{0,}, {0,}};
// int num_obsR[140][140] = {{0,}, {0,}};
int num_obsL[175][175] = {{0,}, {0,}};
int num_obsR[175][175] = {{0,}, {0,}};

int num_freeL[175][175] = {{0,}, {0,}};
int num_freeR[175][175] = {{0,}, {0,}};

// for checking DR error
struct CARPOSE {
    double x,y,th,vel;
};CARPOSE m_car;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_avm_data(new pcl::PointCloud<pcl::PointXYZRGB>);
bool m_flagDR = false;
unsigned int m_avmDRsize = 0, m_DRcnt = 0;
unsigned int CNTforIMAGE_save = 172, CNTforIMAGE = 0;

void arr2real(int recvX, int recvY, double& outX, double& outY) {
    outX = recvX * m_gridResol - (m_gridResol*m_gridDim - m_gridResol) / 2.0;
    outY = recvY * m_gridResol - (m_gridResol*m_gridDim - m_gridResol) / 2.0;
}

void real2arr(double recvX, double recvY, int& outX, int& outY) {
    outX = (m_gridDim/2.0) + recvX / m_gridResol;
    outY = (m_gridDim/2.0) + recvY / m_gridResol;
}

//Yangwoo===============================================================
double GOAL_G[3] = {0.0, 0.0, (0)};
int modx1 = 0, mody1 =0, modx2 = 0, mody2 =0, modx3 = 0, mody3 =0, modx4 = 0, mody4 =0;
int modx1free = 0, mody1free =0, modx2free = 0, mody2free =0, modx3free = 0, mody3free =0, modx4free = 0, mody4free =0;

void coord(double dx, double dy, double xx, double yy, double thh, int& x_, int& y_) {
    double modx = cos(M_PI/2+thh)*dx - sin(M_PI/2+thh)*dy + xx;
    double mody = sin(M_PI/2+thh)*dx + cos(M_PI/2+thh)*dy + yy;
    real2arr(modx, mody, x_, y_);
}

int withinpoint(int x, int y) {
    typedef boost::geometry::model::d2::point_xy<int> point_type;
    typedef boost::geometry::model::polygon<point_type> polygon_type;

    polygon_type poly;
    poly.outer().assign({
        point_type {modx1, mody1}, point_type {modx2, mody2},
        point_type {modx3, mody3}, point_type {modx4, mody4},
        point_type {modx1, mody1}
    });

    point_type p(x, y);

    return boost::geometry::within(p, poly);
}

int withinpoint_free(int x, int y) {
    typedef boost::geometry::model::d2::point_xy<int> point_type;
    typedef boost::geometry::model::polygon<point_type> polygon_type;

    polygon_type poly;
    poly.outer().assign({
        point_type {modx1free, mody1free}, point_type {modx2free, mody2free},
        point_type {modx3free, mody3free}, point_type {modx4free, mody4free},
        point_type {modx1free, mody1free}
    });

    point_type p(x, y);

    return boost::geometry::within(p, poly);
}

void CallbackParkingGoal(const geometry_msgs::PoseArray::ConstPtr& end) {    //[end] which is the coordinates of the goal
    GOAL_G[0] = end->poses[0].position.x;
    GOAL_G[1] = end->poses[0].position.y;
    GOAL_G[2] = tf::getYaw(end->poses[0].orientation);

    coord(M_SPACE_WIDTH/2, -M_SPACE_LENGTH/2,  GOAL_G[0],GOAL_G[1], GOAL_G[2], modx1, mody1);
    coord(M_SPACE_WIDTH/2, M_SPACE_LENGTH/2,   GOAL_G[0],GOAL_G[1], GOAL_G[2], modx2, mody2);
    coord(-M_SPACE_WIDTH/2, M_SPACE_LENGTH/2,  GOAL_G[0],GOAL_G[1], GOAL_G[2], modx3, mody3);
    coord(-M_SPACE_WIDTH/2, -M_SPACE_LENGTH/2, GOAL_G[0],GOAL_G[1], GOAL_G[2], modx4, mody4);

    coord( M_SPACE_WIDTH/2*FREE_ERASE_GAIN, -M_SPACE_LENGTH/2*FREE_ERASE_GAIN, GOAL_G[0],GOAL_G[1], GOAL_G[2], modx1free, mody1free);
    coord( M_SPACE_WIDTH/2*FREE_ERASE_GAIN,  M_SPACE_LENGTH/2*FREE_ERASE_GAIN, GOAL_G[0],GOAL_G[1], GOAL_G[2], modx2free, mody2free);
    coord(-M_SPACE_WIDTH/2*FREE_ERASE_GAIN,  M_SPACE_LENGTH/2*FREE_ERASE_GAIN, GOAL_G[0],GOAL_G[1], GOAL_G[2], modx3free, mody3free);
    coord(-M_SPACE_WIDTH/2*FREE_ERASE_GAIN, -M_SPACE_LENGTH/2*FREE_ERASE_GAIN, GOAL_G[0],GOAL_G[1], GOAL_G[2], modx4free, mody4free);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

void Local2Global(double Lx, double Ly, double &gX, double &gY) {
    gX = m_car.x + (Lx * cos(m_car.th) - Ly * sin(m_car.th));
    gY = m_car.y + (Lx * sin(m_car.th) + Ly * cos(m_car.th));
}

void seg2rgb(cv::Mat input_img, cv::Mat& output_img, cv::Mat& output_img_gray) {
    for (int i = 0 ; i < input_img.rows ; i++) {
        for (int j = 0 ; j < input_img.cols ; j++) {
            // if ((int)input_img.at<uchar>(i, j) != 0 && 
            //     (int)input_img.at<uchar>(i, j) != 1 && 
            //     (int)input_img.at<uchar>(i, j) != 2 && 
            //     (int)input_img.at<uchar>(i, j) != 3 &&
            //     (int)input_img.at<uchar>(i, j) != 4 && 
            //     (int)input_img.at<uchar>(i, j) != 5 &&
            //     (int)input_img.at<uchar>(i, j) != 6 &&
            //     (int)input_img.at<uchar>(i, j) != 9 &&
            //     (int)input_img.at<uchar>(i, j) != 10 &&
            //     (int)input_img.at<uchar>(i, j) != 13 &&
            //     (int)input_img.at<uchar>(i, j) != 14 &&
            //     (int)input_img.at<uchar>(i, j) != 15)
            //     cout << (int)input_img.at<uchar>(i, j) << " " ;
// Segmentation: 14 classes
// (ID and description)
// 0 : background
// 1 : lane - solid, dashed, dotted
// 2 : lane - parking, stop, arrow, etc
// 4 : vehicle - all types
// 6 : wheel
// 9 : general - cone, curbstone, parking block, etc
// 10: cycle, bicyclist, motorcyclist
// 14: pedestrian
// 15: freespace
// 17: parking space
// 18: crosswalk
// 19: speed bump
// 20: foot
// 21: head
            if ((int)input_img.at<uchar>(i, j) == 4 || (int)input_img.at<uchar>(i, j) == 6 || (int)input_img.at<uchar>(i, j) == 17) {
                output_img_gray.at<uchar>(i, j) = 255;
                if ((int)input_img.at<uchar>(i, j) == 17) {
                    output_img_gray.at<uchar>(i, j) = 17;
                }
            }
            else
                output_img_gray.at<uchar>(i, j) = 0;

            switch((int)input_img.at<uchar>(i, j)){
                case 0 : // 
                    output_img.at<Vec3b>(i, j)[0] = 78;
                    output_img.at<Vec3b>(i, j)[1] = 56;
                    output_img.at<Vec3b>(i, j)[2] = 24;
                break; 
                case 1 : // vehicle
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 0;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 2 : // wheel
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 0;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 3 : 
                    output_img.at<Vec3b>(i, j)[0] = 255;
                    output_img.at<Vec3b>(i, j)[1] = 0;
                    output_img.at<Vec3b>(i, j)[2] = 0;
                break;  
                case 4 : 
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 125;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 5 : 
                    output_img.at<Vec3b>(i, j)[0] = 255;
                    output_img.at<Vec3b>(i, j)[1] = 0;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 6 : 
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 255;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 9 : // human
                    output_img.at<Vec3b>(i, j)[0] = 255;
                    output_img.at<Vec3b>(i, j)[1] = 255;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 10 : 
                    output_img.at<Vec3b>(i, j)[0] = 35;
                    output_img.at<Vec3b>(i, j)[1] = 111;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                // case 13 :    // road 
                //     output_img.at<Vec3b>(i, j)[0] = 0;
                //     output_img.at<Vec3b>(i, j)[1] = 255;
                //     output_img.at<Vec3b>(i, j)[2] = 0;
                // break;
                case 14 : 
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 165;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 15 : // cart?
                    output_img.at<Vec3b>(i, j)[0] = 193;
                    output_img.at<Vec3b>(i, j)[1] = 182;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 17 :    // road 
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 255;
                    output_img.at<Vec3b>(i, j)[2] = 0;
                break;
                default :
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 0;
                    output_img.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }
    // cout << endl;
}

void preProcessing(cv::Mat input_img, cv::Mat& output_img)
{
    //resize, bagfile_size: (512, 288) * 3.75 => (1920, 1080)
    cv::resize( input_img, input_img, cv::Size(CROP_ROI_WIDTH, CROP_ROI_HEIGHT), 0, 0, cv::INTER_LINEAR );
    // padding
    cv::Mat temp;

    //padding 1920 1208
    if (Padding_UP)
        cv::copyMakeBorder(input_img, temp, PADDING_VALUE, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
    else
        cv::copyMakeBorder(input_img, temp, 0, PADDING_VALUE, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
    cv::resize( temp, output_img, cv::Size(FULL_IMG_RESOL_WIDTH, FULL_IMG_RESOL_HEIGHT), 0, 0, cv::INTER_LINEAR );
}

void fill_avm_img(cv::Mat input_img, cv::Mat& output_img, int col, int row, 
                    double x_,double y_, double *Dir_param, ocam_model *model, 
                    bool gray, int Direction_Mode){
    XY_coord xy;
    xy = InvProjGND(x_, y_, Dir_param[0], Dir_param[1], Dir_param[2], Dir_param[3], Dir_param[4] ,Dir_param[5], model);
 
    if( ((xy.x < FULL_IMG_RESOL_WIDTH) && (xy.x >= 0)) && ((xy.y < FULL_IMG_RESOL_HEIGHT) && (xy.y >= 0)) ){

        if(gray){
            output_img.at<uint8_t>(int(row), int(col)) = static_cast<uint8_t>(input_img.at<uchar>(xy.y ,xy.x));   //b
            
            int arrX, arrY; real2arr(y_, x_, arrY, arrX);
            if (input_img.at<uchar>(xy.y, xy.x) == 255 || input_img.at<uchar>(xy.y, xy.x) == 17) {
                if (Direction_Mode == LEFT)
                    num_obsL[arrX][arrY]++;
                else if (Direction_Mode == RIGHT)
                    num_obsR[arrX][arrY]++;
            }
            if (input_img.at<uchar>(xy.y, xy.x) == 17)
                if (Direction_Mode == LEFT)
                    num_freeL[arrX][arrY]++;
                else if (Direction_Mode == RIGHT)
                    num_freeR[arrX][arrY]++;
        }
        else{
            output_img.at<cv::Vec3b>(int(row), int(col))[0] = static_cast<uint8_t>(input_img.at<cv::Vec3b>(xy.y, xy.x)[0]);
            output_img.at<cv::Vec3b>(int(row), int(col))[1] = static_cast<uint8_t>(input_img.at<cv::Vec3b>(xy.y, xy.x)[1]);
            output_img.at<cv::Vec3b>(int(row), int(col))[2] = static_cast<uint8_t>(input_img.at<cv::Vec3b>(xy.y, xy.x)[2]);
        }
    }
}

void Inverse_Warping(cv::Mat input_img, cv::Mat& output_img, int Direction_Mode, bool gray) {
    cv::Mat Processed_img;
    preProcessing(input_img, Processed_img);
    if (gray)   output_img = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC1);
    else        output_img = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);

    double x_,y_;

    // Camera Intrinsic & Extrinsic Parameter
    double *Dir_param;
    ocam_model *model;

    if(Direction_Mode == FRONT){
        Dir_param = M_front_param;  model = &front_model;
        for(int row = 0; row < (AVM_IMG_HEIGHT / 2) ; row++){
            for(int col = 0; col < AVM_IMG_WIDTH ; col++){
                // WORLD_coord TO IMG coord
                y_ = (REAL_OCCUPANCY_SIZE_X/2.0) - double (col) * METER_PER_PIXEL;
                x_ = (REAL_OCCUPANCY_SIZE_Y/2.0) - double (row) * METER_PER_PIXEL;

                fill_avm_img(Processed_img, output_img, col, row, x_, y_, Dir_param, model, gray, Direction_Mode);
            }
        }
    }
    else if(Direction_Mode == REAR){
        Dir_param = M_back_param;   model = &rear_model;
        for(int row = AVM_IMG_HEIGHT-1; row > (AVM_IMG_HEIGHT / 2) ; row--){
            for(int col = AVM_IMG_WIDTH-1; col > 0 ; col--){
                // WORLD_coord TO IMG coord
                y_ = -(REAL_OCCUPANCY_SIZE_X/2.0) + double (AVM_IMG_WIDTH - col) * METER_PER_PIXEL;
                x_ = -(REAL_OCCUPANCY_SIZE_Y/2.0) + double (AVM_IMG_WIDTH - row) * METER_PER_PIXEL;

                fill_avm_img(Processed_img, output_img, col, row, x_, y_, Dir_param, model, gray, Direction_Mode);
            }
        }
    }
    else if(Direction_Mode == LEFT){
        Dir_param = M_left_param;   model = &left_model;
        for(int row = AVM_IMG_HEIGHT-1; row> 0 ; row--){
            for(int col = 0; col < (AVM_IMG_WIDTH / 2) ; col++){
                // WORLD_coord TO IMG coord
                y_ = (REAL_OCCUPANCY_SIZE_X/2.0) - double (col) * METER_PER_PIXEL;
                x_ = -(REAL_OCCUPANCY_SIZE_Y/2.0)  + double (AVM_IMG_WIDTH - row) * METER_PER_PIXEL;

                fill_avm_img(Processed_img, output_img, col, row, x_, y_, Dir_param, model, gray, Direction_Mode);
            }
        }
    }
    else if(Direction_Mode == RIGHT){
        Dir_param = M_right_param;  model = &right_model;
        for(int row = 0; row< AVM_IMG_HEIGHT ; row++){
            for(int col = AVM_IMG_WIDTH-1; col > (AVM_IMG_WIDTH / 2) ; col--){
                // WORLD_coord TO IMG coord
                y_ = -(REAL_OCCUPANCY_SIZE_X/2.0) + double (AVM_IMG_WIDTH - col) * METER_PER_PIXEL;
                x_ = (REAL_OCCUPANCY_SIZE_Y/2.0)  - double (row) * METER_PER_PIXEL;

                fill_avm_img(Processed_img, output_img, col, row, x_, y_, Dir_param, model, gray, Direction_Mode);
            }
        }
    }
}

void push_detection_result(const ocamcalib_undistort::ParkingPhantomnetData::ConstPtr& msg, std::vector<DETECTIONS> *det){
    DETECTIONS tmp;
    det->clear();

    // std::cout<< "size() : " << msg->detections.size() << std::endl;

    for (int i = 0 ; i < msg->detections.size() ; i++){
        tmp.classification = msg->detections[i].classification;

        if((tmp.classification == PARKING_SPACE) && (msg->detections[i].probability > OBJ_DET_CONFIDENCE))
        {
            // std::cout<< "tmp.x" << tmp.x<<std::endl;
            tmp.probability = msg->detections[i].probability;
            tmp.x = msg->detections[i].x - ENLARGED_BOX;
            tmp.y = msg->detections[i].y - ENLARGED_BOX;
            tmp.width = msg->detections[i].width + 2*ENLARGED_BOX;
            tmp.height = msg->detections[i].height + 2*ENLARGED_BOX;
            det->push_back(tmp);
        }
    }
    // std::cout<< "size() : " << det->size() << std::endl;
}

// Front
void CallbackPhantom_center(const sensor_msgs::ImageConstPtr& msg) 
{
    cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy(msg, "bgr8" )->image;
    temp1 = cv_frame_resize_pad;
    Inverse_Warping(cv_frame_resize_pad, AVM_front, FRONT, false);
}
// Rear
void CallbackPhantom_rear(const sensor_msgs::ImageConstPtr& msg) 
{
    cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy(msg, "bgr8" )->image;
    Inverse_Warping(cv_frame_resize_pad, AVM_rear, REAR, false);
}
// Left
void CallbackPhantom_left(const sensor_msgs::ImageConstPtr& msg) 
{
    cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy(msg, "bgr8" )->image;
    Inverse_Warping(cv_frame_resize_pad, AVM_left, LEFT, false);
}
// Right
void CallbackPhantom_right(const sensor_msgs::ImageConstPtr& msg) 
{
    cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg, "bgr8" )->image;
    Inverse_Warping(cv_frame_resize_pad, AVM_right, RIGHT, false);
}

//Left Segmentation
void CallbackPhantom_seg_left(const ocamcalib_undistort::ParkingPhantomnetData::ConstPtr& msg) { 
    push_detection_result(msg, &Left_detection);

    cv::Mat cv_frame_seg = cv_bridge::toCvCopy( msg->segmentation, msg->segmentation.encoding )->image;
    cv::Mat cv_frame_raw_new = cv_bridge::toCvCopy( msg->viz, msg->viz.encoding )->image;
    cv::Mat cv_frame_raw_new_gray = cv_bridge::toCvCopy( msg->viz, msg->segmentation.encoding )->image;

    seg2rgb(cv_frame_seg, cv_frame_raw_new, cv_frame_raw_new_gray);
    Inverse_Warping(cv_frame_raw_new, AVM_seg_left, LEFT, false);
    temp2 = cv_frame_raw_new;
    //for occupancy grid map   
    for(int i=0; i < m_gridDim ; i++) for(int j=0 ;j < m_gridDim ;j++) {
        num_obsL[j][i] = 0;
        num_freeL[j][i] = 0;
    }
    Inverse_Warping(cv_frame_raw_new_gray, AVM_seg_left_gray, LEFT, true);
}
//Right Segmentation
void CallbackPhantom_seg_right(const ocamcalib_undistort::ParkingPhantomnetData::ConstPtr& msg) {
    push_detection_result(msg, &Right_detection);
    
    cv::Mat cv_frame_seg = cv_bridge::toCvCopy( msg->segmentation, msg->segmentation.encoding )->image;
    cv::Mat cv_frame_raw_new = cv_bridge::toCvCopy( msg->viz, msg->viz.encoding )->image;
    cv::Mat cv_frame_raw_new_gray = cv_bridge::toCvCopy( msg->viz, msg->segmentation.encoding )->image;

    seg2rgb(cv_frame_seg, cv_frame_raw_new, cv_frame_raw_new_gray);
    Inverse_Warping(cv_frame_raw_new, AVM_seg_right, RIGHT, false);
    temp3 = cv_frame_raw_new;
    for(int i=0; i < m_gridDim ; i++) for(int j=0 ;j < m_gridDim ;j++) {
        num_obsR[j][i] = 0;
        num_freeR[j][i] = 0;
    }
    Inverse_Warping(cv_frame_raw_new_gray, AVM_seg_right_gray, RIGHT, true);
}

void CallbackPhantom_seg_front(const ocamcalib_undistort::ParkingPhantomnetData::ConstPtr& msg) {
    push_detection_result(msg, &Front_detection);
    
    cv::Mat cv_frame_seg = cv_bridge::toCvCopy( msg->segmentation, msg->segmentation.encoding )->image;
    cv::Mat cv_frame_raw_new = cv_bridge::toCvCopy( msg->viz, msg->viz.encoding )->image;
    cv::Mat cv_frame_raw_new_gray = cv_bridge::toCvCopy( msg->viz, msg->segmentation.encoding )->image;
    // temp1 = cv_frame_raw_new;
    seg2rgb(cv_frame_seg, cv_frame_raw_new, cv_frame_raw_new_gray);
    Inverse_Warping(cv_frame_raw_new, AVM_seg_front, FRONT, false);
}

void CallbackPhantom_seg_rear(const ocamcalib_undistort::ParkingPhantomnetData::ConstPtr& msg) {
    push_detection_result(msg, &Rear_detection);

    cv::Mat cv_frame_seg = cv_bridge::toCvCopy( msg->segmentation, msg->segmentation.encoding )->image;
    cv::Mat cv_frame_raw_new = cv_bridge::toCvCopy( msg->viz, msg->viz.encoding )->image;
    cv::Mat cv_frame_raw_new_gray = cv_bridge::toCvCopy( msg->viz, msg->segmentation.encoding )->image;
    temp4 = cv_frame_raw_new;
    seg2rgb(cv_frame_seg, cv_frame_raw_new, cv_frame_raw_new_gray);
    Inverse_Warping(cv_frame_raw_new, AVM_seg_rear, REAR, false);
}

XY_coord BoundingBox_Point(std::vector<DETECTIONS> detection, int direction){
    int Max_Area=0, Max_Area_Cx=0, Max_Area_Cy=0, Max_Width =0, Max_Height=0;
    for(int i=0 ;i < detection.size() ; i++)
    {
        if(detection[i].width * detection[i].height > Max_Area){
            Max_Area = detection[i].width * detection[i].height;
            Max_Area_Cx = detection[i].x;       Max_Area_Cy = detection[i].y;
            Max_Width   = detection[i].width;   Max_Height  = detection[i].height;     
        }
    }
    //1920 (1208 -128 /3.75)
    double *Dir_param;
    ocam_model *model;
    if( direction == REAR ){
        Dir_param = M_back_param;   //extrinsic -> camera pose(rot, trans)
        model = &rear_model;        //intrinsic -> dist_Coeff
    }       
    else if(direction == FRONT ){
        Dir_param = M_front_param;
        model = &front_model; 
    }
    else if( direction == LEFT ){
        Dir_param = M_left_param;
        model = &left_model; 
    }
    else if( direction == RIGHT ){
        Dir_param = M_right_param;
        model = &right_model; 
    }
    
    XY_coord uv;
    
    // std::cout<< "direction : " << direction << std::endl;
    // std::cout<< "Max_Area 중심좌표 " << Max_Area_Cx<< " , " << Max_Area_Cy<< std::endl;

    uv = ProjGND(int(double(Max_Area_Cx + 0.5*Max_Width) * double(RESIZE_BAG_TO_ORG)), int(double(Max_Area_Cy + 0.5*Max_Height) * double(RESIZE_BAG_TO_ORG) + PADDING_VALUE),
                    Dir_param[0], Dir_param[1], Dir_param[2], Dir_param[3], Dir_param[4] ,Dir_param[5], model);

    // std::cout<< "xy.x and y : " << uv.x << " , " << uv.y<< std::endl;
    if(direction ==FRONT)
        circle(temp1, Point(Max_Area_Cx+0.5*Max_Width, Max_Area_Cy+0.5*Max_Height), 2, Scalar(255,0,0), -1);
    if(direction ==LEFT)
        circle(temp2, Point(Max_Area_Cx+0.5*Max_Width, Max_Area_Cy+0.5*Max_Height), 2, Scalar(255,0,0), -1);
    if(direction ==RIGHT)
        circle(temp3, Point(Max_Area_Cx+0.5*Max_Width, Max_Area_Cy+0.5*Max_Height), 2, Scalar(255,0,0), -1);
    if(direction ==REAR)
        circle(temp4, Point(Max_Area_Cx+0.5*Max_Width, Max_Area_Cy+0.5*Max_Height), 2, Scalar(255,0,0), -1);
    // ocamcalib_undistort::Bounding_Box bounding_box;
    // bounding_box.u = uv.x;
    // bounding_box.v = uv.y;

    // Pub_Bounding_Box.publish(bounding_box);
    return uv;
}
void Bounding_Box_pub(){
    XY_coord front_xy = BoundingBox_Point(Front_detection, FRONT);
    XY_coord rear_xy  = BoundingBox_Point(Rear_detection, REAR);
    XY_coord left_xy  = BoundingBox_Point(Left_detection, LEFT);
    XY_coord right_xy = BoundingBox_Point(Right_detection, RIGHT);

    std_msgs::Float32MultiArray msg;

    msg.data.push_back((float)(front_xy.x));   msg.data.push_back((float)(front_xy.y));
    msg.data.push_back((float)(rear_xy.x ));   msg.data.push_back((float)(rear_xy.y ));
    msg.data.push_back((float)(left_xy.x ));   msg.data.push_back((float)(left_xy.y ));
    msg.data.push_back((float)(right_xy.x));   msg.data.push_back((float)(right_xy.y));

    msg.data.push_back((float)(AVM_IMG_WIDTH));
    msg.data.push_back((float)(REAL_OCCUPANCY_SIZE_X));
    
    Pub_Boundingbox.publish(msg);
}

void AVMpointCloud(cv::Mat img) {
    int avmCutRange = 0, idxSparse = 1;
    if (m_flagDR) {idxSparse = 3; avmCutRange = 75;}

    // if (m_DRcnt%3 == 0) {
        m_DRcnt = 0;
        for(int i = avmCutRange ; i < img.size().height - avmCutRange ; i = i+idxSparse){
            for(int j = avmCutRange ; j < img.size().width -avmCutRange ; j = j+idxSparse){
                // if(!(img.at<cv::Vec3b>(i,j)[1] == 0) && !(img.at<cv::Vec3b>(i,j)[0] == 0) && !(img.at<cv::Vec3b>(i,j)[2] == 0)) {   
                    double x = (REAL_OCCUPANCY_SIZE_X / 2) - METER_PER_PIXEL * i, y = (REAL_OCCUPANCY_SIZE_Y/2) - METER_PER_PIXEL * j, gX, gY;
                    Local2Global(x, y, gX, gY);

                    pcl::PointXYZRGB pt;
                    pt.x = gX;  pt.y = gY;  pt.z = 0.0;

                    uint8_t r = static_cast<uint8_t>(img.at<cv::Vec3b>(i,j)[2]), g = static_cast<uint8_t>(img.at<cv::Vec3b>(i,j)[1]), b = static_cast<uint8_t>(img.at<cv::Vec3b>(i,j)[0]);
                    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                    pt.rgb = *reinterpret_cast<float*>(&rgb);

                    m_avm_data->push_back(pt);
                // }
            }
        }
    // }

    if (m_flagDR) {
        if (m_avm_data->size() > 500000)
            m_avm_data->erase(m_avm_data->begin(), m_avm_data->begin() + 10000);
    } else {
        m_avm_data->erase(m_avm_data->begin(), m_avm_data->begin() + m_avmDRsize);
        m_avmDRsize = m_avm_data->size();
    }
    
    // cout << m_avm_data->size() << endl;
    Pub_AVM_DR.publish(m_avm_data);
}

void CallbackPhantom_DR(const std_msgs::Float32MultiArray::ConstPtr& msg){
    m_car.x = msg->data.at(0);      // x
    m_car.y = msg->data.at(1);      // y
    m_car.th = msg->data.at(2);     // theta
    m_car.vel = msg->data.at(3);    // [m/s]
    m_flagDR = true;
    m_DRcnt++;

    AVMpointCloud(aggregated_side_img);
}

void occupancyGridmapPub() {
    int cnt = 0;
    // occupancyGridMap.data.clear();
    occupancyGridMap.data.resize(occupancyGridMap.info.width*occupancyGridMap.info.width);
    for(int i=0; i < m_gridDim ; i++) 
        for(int j=0 ;j < m_gridDim ; j++) {
            if (num_obsL[j][i] > 0 || num_obsR[j][i] > 0)   {
                occupancyGridMap.data[cnt] = 100;

                if (withinpoint(j, i) == 1)
                    occupancyGridMap.data[cnt] = 0;
            }
            else                                            
                occupancyGridMap.data[cnt] = 0;

            if (num_freeL[j][i] > 0 || num_freeR[j][i] > 0)   {
                if (withinpoint_free(j, i) == 1)
                    occupancyGridMap.data[cnt] = 0;
            }
            cnt++;
        }

    if (Pub_occupancyGridMap.getNumSubscribers() > 0)
        Pub_occupancyGridMap.publish(occupancyGridMap);
}

int main(int argc, char **argv)
{   
    ros::init(argc, argv, "undistort_node");
    ros::NodeHandle nodeHandle("~");  

    // Load Intrinsic Parameter from Directory**
    if(flag == 0) {
        if(!get_ocam_model(&front_model, calibration_front.c_str()) ||
        !get_ocam_model(&left_model, calibration_left.c_str()) ||
        !get_ocam_model(&right_model, calibration_right.c_str()) ||
        !get_ocam_model(&rear_model, calibration_rear.c_str()))
            return 2;
        flag =1;
    }

    Sub_phantom_side_left       = nodeHandle.subscribe("/csi_cam/side_left/image_raw", 1, CallbackPhantom_left);
    Sub_phantom_side_right      = nodeHandle.subscribe("/csi_cam/side_right/image_raw", 1, CallbackPhantom_right);
    Sub_phantom_left_seg    = nodeHandle.subscribe("/parking/phantomnet/side_left", 1 , CallbackPhantom_seg_left);
    Sub_phantom_right_seg   = nodeHandle.subscribe("/parking/phantomnet/side_right", 1 , CallbackPhantom_seg_right);

    Sub_phantom_front_center    = nodeHandle.subscribe("/csi_cam/front_center_svm/image_raw", 1 , CallbackPhantom_center);
    Sub_phantom_rear_center     = nodeHandle.subscribe("/csi_cam/rear_center_svm/image_raw", 1, CallbackPhantom_rear);
    Sub_phantom_front_seg    = nodeHandle.subscribe("/parking/phantomnet/front_center_svm", 1 , CallbackPhantom_seg_front);
    Sub_phantom_rear_seg   = nodeHandle.subscribe("/parking/phantomnet/rear_center_svm", 1 , CallbackPhantom_seg_rear);

    // Sub_phantom_tmp   = nodeHandle.subscribe("/parking/phantomnet/side_right", 1 , Callback_tmp);

    // undistorted Image
    // Sub_phantom_front_seg    = nodeHandle.subscribe("/parking/phantomnet/front_center", 1 , CallbackPhantom_seg_front);
    // Sub_phantom_rear_seg   = nodeHandle.subscribe("/parking/phantomnet/rear_center", 1 , CallbackPhantom_seg_rear);

    // Sub_phantom_DR_Path = nodeHandle.subscribe("/LocalizationData", 1 , CallbackPhantom_DR);
    // Sub_parkingGoal = nodeHandle.subscribe("/parking_cands", 1, CallbackParkingGoal);

    // Pub_AVM_side_img          = nodeHandle.advertise<sensor_msgs::Image>("/AVM_side_image", 1);
    Pub_AVM_side_img          = nodeHandle.advertise<sensor_msgs::Image>("/AVM_image", 1);
    Pub_AVM_center_img          = nodeHandle.advertise<sensor_msgs::Image>("/AVM_center_image", 1);

    Pub_AVM_side_seg_img      = nodeHandle.advertise<sensor_msgs::Image>("/AVM_side_seg_image", 1);
    Pub_AVM_center_seg_img      = nodeHandle.advertise<sensor_msgs::Image>("/AVM_center_seg_image", 1);
    Pub_AVM_seg_img          = nodeHandle.advertise<sensor_msgs::Image>("/AVM_seg_image", 1);
    
    Pub_AVM_seg_img_gray = nodeHandle.advertise<sensor_msgs::Image>("/AVM_seg_image_gray", 1);
    Pub_AVM_DR           = nodeHandle.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/AVM_image_DR", 1);

    // Pub_Bounding_Box     = nodeHandle.advertise<ocamcalib_undistort::Bounding_Box>("/bounding_box", 1);
    Pub_Boundingbox = nodeHandle.advertise<std_msgs::Float32MultiArray>("/boundingbox_point", 1);


    occupancyGridMap.header.frame_id = "map";
    occupancyGridMap.info.resolution = m_gridResol;
    occupancyGridMap.info.width = occupancyGridMap.info.height = m_gridDim;
    occupancyGridMap.info.origin.position.x = occupancyGridMap.info.origin.position.y = -m_dimension/2 - m_gridResol*2;
    occupancyGridMap.info.origin.position.z = 0.1;
    occupancyGridMap.data.resize(occupancyGridMap.info.width*occupancyGridMap.info.width);
    Pub_occupancyGridMap = nodeHandle.advertise<nav_msgs::OccupancyGrid>("/occ_map", 1);

    m_avm_data->clear();
    m_avm_data->header.frame_id = "map";
    
    ros::Rate loop_rate(100);
    // ros::spin();

/////////////////////////////////////////////////////////////////////////////////////////////////////
//@Yangwoo

    m_car.x = m_car.y = m_car.th = 0.0;
    while(ros::ok()) { 
        if(m_flag_both_dir == false){
            if (m_flagDR) {
                ros::AsyncSpinner spinner(4+1);
                spinner.start();
            }else {
                ros::AsyncSpinner spinner(4);
                spinner.start();
            }
        }
        else if(m_flag_both_dir == true){
            if (m_flagDR) {
                ros::AsyncSpinner spinner(8+1);
                spinner.start();
            }else {
                ros::AsyncSpinner spinner(8);
                spinner.start();
            }
        }

        // if(m_flag_dir == 0){
        //     aggregated_side_img = AVM_right + AVM_left;
        //     aggregated_side_seg_img = AVM_seg_right + AVM_seg_left;
        //     aggregated_seg_img_gray = AVM_seg_right_gray + AVM_seg_left_gray;
        // }
        // else if(m_flag_dir == 1){
        //     aggregated_center_img = AVM_front + AVM_rear;
        //     aggregated_center_seg_img = AVM_seg_front + AVM_seg_rear;
        // }
        if(m_flag_dir == 2){
            aggregated_side_img = AVM_right + AVM_left;
            aggregated_side_seg_img = AVM_seg_right + AVM_seg_left;

            aggregated_center_img = AVM_front + AVM_rear;       //  AVM_rear;
            aggregated_center_seg_img = AVM_seg_front + AVM_seg_rear;             // AVM_seg_rear;

            aggregated_seg_img_gray = AVM_seg_right_gray + AVM_seg_left_gray;
        }

        for(int i=0 ;i < Front_detection.size() ; i++)
        {
            int label =Front_detection[i].classification;
            rectangle(temp1, Rect( Front_detection[i].x, Front_detection[i].y, Front_detection[i].width, Front_detection[i].height ), 
                Scalar(10*label % 256 , 20*label % 256, 30*label %256), 2, 8, 0);
        }
        for(int i=0 ;i < Left_detection.size() ; i++)
        {
            int label =Left_detection[i].classification;
            rectangle(temp2, Rect( Left_detection[i].x, Left_detection[i].y, Left_detection[i].width, Left_detection[i].height ), 
                Scalar(10*label % 256 , 20*label % 256, 30*label %256), 2, 8, 0);
        }
        for(int i=0 ;i < Right_detection.size() ; i++)
        {
            int label =Right_detection[i].classification;
            rectangle(temp3, Rect( Right_detection[i].x, Right_detection[i].y, Right_detection[i].width, Right_detection[i].height ), 
                Scalar(10*label % 256 , 20*label % 256, 30*label %256), 2, 8, 0);
        }
        for(int i=0 ;i < Rear_detection.size() ; i++)
        {
            int label =Rear_detection[i].classification;
            rectangle(temp4, Rect( Rear_detection[i].x, Rear_detection[i].y, Rear_detection[i].width, Rear_detection[i].height ), 
                Scalar(10*label % 256 , 20*label % 256, 30*label %256), 2, 8, 0);
        }

        // imwrite("/home/dyros-phantom/catkin_ws/src/generalized_hough/include/rear_image.png", aggregated_center_img);

        // if (CNTforIMAGE++ % 3 == 0)
        //     imwrite("/home/dyros-phantom/catkin_ws/src/ocamcalib_undistort/image/PhantomAVM_"+to_string(CNTforIMAGE_save++)+".jpg", aggregated_img);
    
        Bounding_Box_pub();

        Pub_AVM_side_img.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", aggregated_side_img).toImageMsg());
        Pub_AVM_center_img.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", aggregated_center_img).toImageMsg());

        Pub_AVM_side_seg_img.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", aggregated_side_seg_img).toImageMsg());
        Pub_AVM_center_seg_img.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", aggregated_center_seg_img).toImageMsg());

        Pub_AVM_seg_img_gray.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", aggregated_seg_img_gray).toImageMsg());

        occupancyGridmapPub();

        if (!m_flagDR)
            AVMpointCloud(aggregated_side_img);

        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}
