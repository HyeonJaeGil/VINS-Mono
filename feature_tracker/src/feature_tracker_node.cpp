#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData;
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;


void img_mask_callback(const sensor_msgs::ImageConstPtr &img_msg,
                        const sensor_msgs::ImageConstPtr &mask_msg)
{
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();
    // frequency control
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat show_img = ptr->image;

    // handle mask msg
    cv_bridge::CvImageConstPtr mask_ptr;
    sensor_msgs::Image mask_img;
    mask_img.header = mask_msg->header;
    mask_img.height = mask_msg->height;
    mask_img.width = mask_msg->width;
    mask_img.is_bigendian = mask_msg->is_bigendian;
    mask_img.step = mask_msg->step;
    mask_img.data = mask_msg->data;
    mask_img.encoding = "mono8";
    mask_ptr = cv_bridge::toCvCopy(mask_img, sensor_msgs::image_encodings::MONO8);


    TicToc t_r;
    trackerData.readMask(mask_ptr->image);
    trackerData.readImage(ptr->image, img_msg->header.stamp.toSec());
#if SHOW_UNDISTORTION
    trackerData.showUndistortion("undistortion");
#endif
    for (unsigned int i = 0;; i++)
        if (!trackerData.updateID(i))
            break;

   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "camera";

        auto &un_pts = trackerData.cur_un_pts;
        auto &cur_pts = trackerData.cur_pts;
        auto &ids = trackerData.ids;
        auto &pts_velocity = trackerData.pts_velocity;
        for (unsigned int i = 0; i < ids.size(); i++)
        {
            if (trackerData.track_cnt[i] > 1)
            {
                int p_id = ids[i];
                geometry_msgs::Point32 p;
                p.x = un_pts[i].x;
                p.y = un_pts[i].y;
                p.z = 1;

                feature_points->points.push_back(p);
                id_of_point.values.push_back(p_id);
                u_of_point.values.push_back(cur_pts[i].x);
                v_of_point.values.push_back(cur_pts[i].y);
                velocity_x_of_point.values.push_back(pts_velocity[i].x);
                velocity_y_of_point.values.push_back(pts_velocity[i].y);
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);

        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            cv::Mat tmp_img = ptr->image;
            cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

            for (unsigned int i = 0; i < trackerData.cur_pts.size(); i++)
            {
                double len = std::min(1.0, 1.0 * trackerData.track_cnt[i] / WINDOW_SIZE);
                cv::circle(tmp_img, trackerData.cur_pts[i], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
            }
            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_DEBUG("whole feature tracker processing costs: %f", t_r.toc());
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    trackerData.readIntrinsicParameter(CAM_NAMES);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Subscriber<sensor_msgs::Image> img_sub(n, IMAGE_TOPIC, 1);
    message_filters::Subscriber<sensor_msgs::Image> mask_sub(n, MASK_TOPIC, 1);
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), img_sub, mask_sub);
    sync.registerCallback(boost::bind(&img_mask_callback, _1, _2));

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?