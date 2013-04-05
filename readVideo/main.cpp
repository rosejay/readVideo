#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include "opencv2/imgproc/imgproc.hpp"

#define PI 3.14159265

using std::vector;
using namespace std;
using namespace cv;

struct myLines {
    CvPoint p1;
    CvPoint p2;
} ;

struct myData {
    int type; // 1: Fixation  2: Saccade
    CvPoint start;
    CvPoint end;
    CvPoint calstart;
    CvPoint calend;
    CvPoint rezstart;
    CvPoint rezend;
    float duration;
    float resize;
    int frame;
} ;

struct myRect{
    int x;
    int y;
    int width;
    int height;
} ;

// setting
int myThreshold = 160;  // 80 160
int rectType = 1; // 1: horizontal    2: vertical
int divCalib = 180;
// data
myRect focusArea;
vector<int> framePoint;
vector<myData> points;
// img
Mat outputImg;
vector<Mat> calibratedImg;
int fps;
int dstWidth, dstHeight, resizeWidth, resizeHeight;
bool isFirst = true;
vector<myLines> vLines;
vector<CvPoint> interPoint;
CvPoint leftTop, rightTop, leftBottom, rightBottom;
CvPoint tempLeftTop, tempRightTop, tempLeftBottom, tempRightBottom;
int imgWidth, imgHeight;

bool file_exists(const char * filename){
    if (FILE * file = fopen(filename, "r"))
    {
        fclose(file);
        return true;
    }
    return false;
}
Mat drawAllData(Mat img){
    
    for(int i = 0; i<points.size(); i++){
        
        int col = 255 - i;
        int radius = points.at(i).duration * 100;
        if(points.at(i).type == 1){
            
            circle(  img, points.at(i).rezstart, radius, cvScalar(col,col,col), -1 );
        }
        else if(points.at(i).type == 2){
            circle(  img, points.at(i).rezstart, radius, cvScalar(col,col,col), -1 );
            line(img, points.at(i).rezstart, points.at(i).rezend, cvScalar(col,col,col), 3);
            circle(  img, points.at(i).rezend, radius, cvScalar(col,col,col), -1 );
        }
        
    }
    return img;
    
}

/*
 * filter
 * get the distance between two points
 */
double pointsDistance(CvPoint p1, CvPoint p2){
    
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}
/*
 * filter
 * get the distance between two points
 */
void filterPoint(int index, int type){
    
    if(type == 1){
        // fix fix
        points.at(index).start.x = ( points.at(index-1).start.x + points.at(index-2).start.x  + points.at(index-3).start.x + points.at(index-4).start.x )/4;
        points.at(index).start.y = ( points.at(index-1).start.y + points.at(index-2).start.y  + points.at(index-3).start.y + points.at(index-4).start.y )/4;
    }
    else if(type == 2){
        // sac fix
        points.at(index).start.x = ( points.at(index-1).end.x + points.at(index-1).start.x + points.at(index-2).start.x  + points.at(index-3).start.x )/2;
        points.at(index).start.y = ( points.at(index-1).end.y + points.at(index-1).start.y + points.at(index-2).start.y  + points.at(index-3).start.y )/2;
    }
    else if(type == 3){
        // fix sac 
        points.at(index).end.x = ( points.at(index).start.x + points.at(index-1).start.x + points.at(index-2).start.x  + points.at(index-3).start.x )/2;
        points.at(index).end.y = ( points.at(index).start.y + points.at(index-1).start.y + points.at(index-2).start.y  + points.at(index-3).start.y )/2;
    }
    else if(type == 4){
        // sac sac
        points.at(index).end.x = ( points.at(index).start.x + points.at(index-1).start.x + points.at(index-1).end.x + points.at(index-2).start.x )/2;
        points.at(index).end.y = ( points.at(index).start.y + points.at(index-1).start.y + points.at(index-1).end.y + points.at(index-2).start.y )/2;
    }
    
}

/*
 * calibrate 2D point by the calculated matrix
 */
CvPoint calibratePoint(Mat m, CvPoint p){

    // start calibrate point
    Mat sample = (Mat_<double>(3,1)<< p.x, p.y, 1); // x, y
    Mat r = m*sample;  //变换矩阵乘以样例点，得到结果点的其次坐标
    double s = r.at<double>(2,0);  //归一化尺度

    p.x = ceil(r.at<double>(0,0)/s);
    p.y = ceil(r.at<double>(1,0)/s);
    
    //cout << p.x << " " << p.y << " cal" << endl;
    return p;
}

/*
 * check if focused inside a rect
 */
float focusTime = 0;
void checkFocusRect(myData d){
    
    focusArea.x = 200;
    focusArea.y = 200;
    focusArea.width = 200;
    focusArea.height = 200;
    
    //cout << d.duration << endl;
    if( d.start.x<focusArea.x + focusArea.width && d.start.x>focusArea.x
       && d.start.y<focusArea.height + focusArea.y && d.start.y>focusArea.y )
        focusTime += d.duration;
    
    //printf("%f, %d\n", focusTime, d.frame);
}

/*
 * p1 p2 on line1
 * o1 o2 on line2
 * calculate the inter point of line1 line2
 */
CvPoint segmentsIntr(CvPoint p1, CvPoint p2, CvPoint o1, CvPoint o2){
    
    // init value
    long a1 = p2.y - p1.y;
    long b1 = p1.x - p2.x;
    long c1 = b1*p1.y + a1*p1.x;
    
    long a2 = o2.y - o1.y;
    long b2 = o1.x - o2.x;
    long c2 = b2*o1.y + a2*o1.x;
    
    
    if(b2 == 0 && b1 == 0){
        
        return cvPoint(-1, -1);
    }
    else if(b2 == 0){
        
        long x = o1.x;
        long y = (int)(c1-a1*x)/b1;
        //cout << o1.x  << " dd " << o2.x << " dd " << y << endl;
        return cvPoint((int)x, (int)y);
    }
    else if(b1 == 0){
        
        long x = p1.x;
        long y = (int)(c2-a2*x)/b2;
        //cout << p1.x  << " dd " << p2.x << "dd" << y << endl;
        return cvPoint((int)x, (int)y);
    }
    else{
        
        double k1 = -a1/b1;
        double k2 = -a2/b2;
        
        // if intersection angle < 10
        double result = atan(abs((k2-k1)/(1+k1*k2))) * 180 / PI;
        if(result<10)
            return cvPoint(-1, -1);
        
        // if denominator == 0 parallel
        long denominator = a1*b2 - a2*b1;
        if (denominator==0) {
            return cvPoint(-1, -1);
        }
        
        // intersection point(x,y)
        long x = (b2*c1-b1*c2) / denominator ;
        long y = (a1*c2-a2*c1) / denominator;
        
        //cout << x << " " << y << " " << p1.x << " " << p2.x << " " << o1.x << " " << o2.x << " " << endl;
        // if the point is inside the image
        if ( x>0 && x<imgWidth&& y>0&& y<imgHeight ){
            // return p
            return cvPoint((int)x,(int)y);
        }
        // or no intersection
        return cvPoint(-1, -1);
    }
    
    
}

/*
 * check from all interpoint to get four cornered point
 */
void checkPoint(int width, int height){
    
    
    tempLeftTop = tempRightTop = tempLeftBottom = tempRightBottom = cvPoint(width, height);
    int x,y;
    CvPoint p;
    
    for ( int i = 0; i<interPoint.size(); i++) {
        p = interPoint.at(i);
        x = p.x;
        y = p.y;
        
        //cout << x << "&" << y << endl;
        if( x>0 && y>0 && x<(imgWidth-divCalib) && y<imgHeight ){
            
            if(x<tempLeftTop.x && y<tempLeftTop.y)
                tempLeftTop = p;
            else if(x<tempLeftBottom.x && y>tempLeftBottom.y)
                tempLeftBottom = p;
            else if(x>tempRightTop.x && y<tempRightTop.y)
                tempRightTop = p;
            else if(x>tempRightBottom.x && y>tempRightBottom.y)
                tempRightBottom = p;

        }
                
    }
    //printf("temp, %d %d %d %d %d %d %d %d\n",tempLeftTop.x, tempLeftTop.y, tempRightTop.x, tempRightTop.y,
           //tempRightBottom.x, tempRightBottom.y, tempLeftBottom.x, tempLeftBottom.y);
    
    // if some frame is not complete
    // don't change the four points
    
    bool isBroken = false;
    
    if(tempLeftBottom.x == width || tempLeftBottom.y == height)
        isBroken = true;
    if(tempLeftTop.x == width || tempLeftTop.y == height)
        isBroken = true;
    if(tempRightBottom.x == width || tempRightBottom.y == height)
        isBroken = true;
    if(tempRightTop.x == width || tempRightTop.y == height)
        isBroken = true;
    
    
    if (!isBroken) {
        leftTop.operator=(tempLeftTop);
        leftBottom.operator=(tempLeftBottom);
        rightTop.operator=(tempRightTop);
        rightBottom.operator=(tempRightBottom);
    }
    
}

/*
 * caliberation
 */
Mat transformImg( IplImage* img, int index ){
    
    interPoint.clear();
    vLines.clear();
    
    IplImage *gray = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    IplImage *smooth = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    
    
    //Step 1 - Convert from RGB to grayscale (cvCvtColor)
    cvCvtColor(img, gray, CV_RGB2GRAY);
    //printf("%d %d \n",cvGetSize(img).width, cvGetSize(img).height);
    
    //Step 2 - Smooth (cvSmooth)
    cvSmooth( gray, smooth, CV_BLUR, 9, 9, 2, 2);
    
    //Step 3 - cvThreshold
    cvThreshold(gray,gray, myThreshold, 255, CV_THRESH_BINARY);
    
    //Step 4 - Dilate and Erode
    cvDilate(gray, gray, NULL, 8); //dilation (3*3 kernel)
    cvErode(gray, gray, NULL, 8);
    
    //Step 5 - Detect edges (cvCanny)
    cvCanny( gray, gray, 50, 150, 3 );
    
    
    //Step 6 - Hough transform
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* lines = 0;
    lines = cvHoughLines2( gray, storage, CV_HOUGH_STANDARD, 1, CV_PI/180, 75, 0, 0 );
    cvZero( gray );
    
    //Step 7 - draw lines on original &img
    for( int i = 0; i < MIN(lines->total,100); i++ )
    {
        float* line = (float*)cvGetSeqElem(lines,i);
        float rho = line[0];
        float theta = line[1];
        CvPoint pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        
        // push data
        myLines ml;
        ml.p1 = pt1;
        ml.p2 = pt2;
        vLines.push_back(ml);
        
        //cvLine( img, pt1, pt2, CV_RGB(100,100,100), 1, 8, 0);
        //printf("%d %d %d %d\n",pt1.x,pt1.y,pt2.x,pt2.y);
    }
    //cvNamedWindow("show1",1);
    //cvShowImage("show1",img);

    
    CvPoint p;
    for( int i = 0; i < MIN(lines->total,100); i++ ){
        for( int j = i+1; j < MIN(lines->total,100); j++ ){
            
            
            p = segmentsIntr(vLines.at(i).p1, vLines.at(i).p2, vLines.at(j).p1, vLines.at(j).p2);
            
            // if it is an intersection point
            if(p.x!=-1 && p.y!=-1)
                interPoint.push_back(p);
            
            //printf("%d %d %d %d\n",p.x,p.y,i,j);
        }
    }
    
    // get leftTop, leftBottom, rightTop, rightBottom
    checkPoint(cvGetSize(img).width/2, cvGetSize(img).height/2);
    printf("lll, %d %d %d %d %d %d %d %d\n",leftTop.x, leftTop.y, rightTop.x, rightTop.y, rightBottom.x, rightBottom.y, leftBottom.x, leftBottom.y);
    
    
    //return img;
    Point2f src_vertices[4];
    
    src_vertices[0].x = leftTop.x;
    src_vertices[0].y = leftTop.y;
    src_vertices[1].x = rightTop.x;
    src_vertices[1].y = rightTop.y;
    src_vertices[2].x = rightBottom.x;
    src_vertices[2].y = rightBottom.y;
    src_vertices[3].x = leftBottom.x;
    src_vertices[3].y = leftBottom.y;
    
    // define dstWidth & dstHeight, dstHeight = dstWidth/4*3;
    (rightTop.x-leftTop.x)>(rightBottom.x-leftBottom.x)?dstWidth = (rightTop.x-leftTop.x):dstWidth = (rightBottom.x-leftBottom.x);
    
    if(rectType == 1)
        dstHeight = dstWidth/4*3;
    else if(rectType == 2)
        dstHeight = dstWidth/3*4;
    
    
    if(isFirst){
        resizeWidth = dstWidth;
        resizeHeight = dstHeight;
    }
    
    
    
    //printf("%d,%d\n",dstWidth, dstHeight);
    
    Point2f dst_vertices[4];
    dst_vertices[0] = Point(0, 0);
    dst_vertices[1] = Point(dstWidth, 0);
    dst_vertices[2] = Point(dstWidth, dstHeight);
    dst_vertices[3] = Point(0, dstHeight);
    
    Mat warpAffineMatrix = getPerspectiveTransform(src_vertices, dst_vertices);
    //std::cout << "M = "<< std::endl << " " << warpAffineMatrix << std::endl << std::endl;
    //cout << "M = " << warpAffineMatrix << std::endl;
    
    //Mat src = imread("/Users/lin/Desktop/test2.jpg", 1);
    
    Mat src = cvarrToMat(img);
    
    //printf("%d %d\n",box.boundingRect().width, box.boundingRect().height);
    
    //std::cout << "src = "<< src.size() << std::endl;
    
    Mat rotated, newImg;
    Size size(dstWidth, dstHeight);
    
    //std::cout << "src = "<< size.width << std::endl << size.height << std::endl;
    
    warpPerspective(src, rotated, warpAffineMatrix, size);
    
    if(isFirst){
        outputImg = rotated;
        isFirst = false;
    }
    
    // resize image
    //cout << resizeWidth + "  " + resizeHeight << endl;
    resize(rotated, newImg, cvSize(resizeWidth, resizeHeight));
    
    
    //cout<< index <<" index "<<points.at(index).start.x << " " << points.at(index).start.y << "ddddd" << endl;
    
    // calibrate point
    if(points.at(index).type == 1){
        points.at(index).calstart = calibratePoint(warpAffineMatrix, points.at(index).start);
    }
    else if(points.at(index).type == 2){
        points.at(index).calstart = calibratePoint(warpAffineMatrix, points.at(index).start);
        points.at(index).calend = calibratePoint(warpAffineMatrix, points.at(index).end);
    }
    
    // set resize value
    points.at(index).resize = resizeWidth * 1.0 / dstWidth;
    points.at(index).rezstart = cvPoint(points.at(index).calstart.x * points.at(index).resize,
                                        points.at(index).calstart.y * points.at(index).resize);
    points.at(index).rezend = cvPoint(points.at(index).calend.x * points.at(index).resize,
                                        points.at(index).calend.y * points.at(index).resize);
    
    //draw
    int radius = points.at(index).duration * 100;
    if(points.at(index).type == 1){
        
        circle(  newImg, points.at(index).calstart, radius, cvScalar(0,0,0), -1 );
    }
    else if(points.at(index).type == 2){
        
        circle(  newImg, points.at(index).calstart, radius, cvScalar(0,0,0), -1 );
        line(  newImg, points.at(index).calstart, points.at(index).calend, cvScalar(0,0,0), 3);
        circle(  newImg, points.at(index).calend, radius, cvScalar(0,0,0), -1 );
    }
    
    calibratedImg.push_back(newImg);
    
    return newImg;
    
}






/*
 * read data from txt
 */
void readData(){
    
    int frameTime = 0; // time line, frame number
    //int fps = 30;
    // Open your file
    ifstream someStream( "/Users/lin/Desktop/2903-1.txt" );
    
    // Set up a place to store our data read from the file
    string line;
    
    // Read and throw away the first line simply by doing
    // nothing with it and reading again
    getline( someStream, line );
    
    long startF = 0;
    // Now begin your useful code
    while( !someStream.eof() ) {
        // This will just over write the first line read
        getline( someStream, line );
        
        
        if(line.substr(0,7) == "Saccade" || line.substr(0,8) == "Fixation"){
            
            // split data
            vector<int> arr;
            istringstream iss(line);
            do{
                string sub;
                iss >> sub;
                arr.push_back(atoi(sub.c_str()));
                //cout << sub << endl;
            } while (iss);
            
            
            
            if(!startF){
                startF = arr.at(4);
            }
            
            frameTime = arr.at(5) - startF;
            
                
            //cout << frameTime << endl;
            
            // add data
            framePoint.push_back(frameTime * fps / 1000000);
            myData data;
            data.duration = (arr.at(5) - arr.at(4)) / 1000000.0;
            //cout << data.duration << endl;
            data.frame = frameTime * fps / 1000000;
            
            if(line.substr(0,8) == "Fixation"){

                data.type = 1;
                data.start = cvPoint(arr.at(7), arr.at(8));
                
                
            }
            else if(line.substr(0,7) == "Saccade"){
                
                data.type = 2;
                data.start = cvPoint(arr.at(7), arr.at(8));
                data.end = cvPoint(arr.at(9), arr.at(10));
            }
            
            // get a continuous path
            int len = (int)points.size();
            int index = len - 1;
            int block = 2; // L for filter
            int disThreshold = 150;
            /*
            // filter
            if(len >= 5){
            
                // fix & fix
                if(data.type == 1 && points.at(index).type == 1){
                    
                    // filter
                    if(disThreshold < pointsDistance(points.at(index-1).start, points.at(index).start)){
                        filterPoint(index, 1);
                        cout << "filter" << endl;
                    }
                    
                }
                // sac & fix || sac & sac
                else if((data.type == 1 && points.at(index).type == 2)
                        || (data.type == 2 && points.at(index).type == 2)){
                    
                    // filter
                    if(disThreshold < pointsDistance(points.at(index-1).end, points.at(index).start)){
                        filterPoint(index, 2);
                        cout << "filter" << endl;
                        
                    }
                    
                    // if sac & sac
                    if(data.type == 2 && points.at(index).type == 2){
                        
                        // filter
                        if(disThreshold < pointsDistance(points.at(index).end, points.at(index).start)){
                            filterPoint(index, 4);
                            cout << "filter" << endl;
                        }
                    }
                    
                }
                // fix & sac
                else if(data.type == 2 && points.at(index).type == 1){
                    
                    // filter
                    if(disThreshold < pointsDistance(points.at(index-1).start, points.at(index).start)){
                        filterPoint(index, 1);
                        cout << "filter" << endl;
                        
                    }
                    
                    if(disThreshold < pointsDistance(points.at(index).start, points.at(index).end)){
                        filterPoint(index, 3);
                        cout << "filter" << endl;
                        
                    }
                }
            }
            */
            // set value
            if(len){
                
                // fix & fix
                if(data.type == 1 && points.at(index).type == 1){
                    
                    // nothing
                }
                // sac & fix || sac & sac
                else if((data.type == 1 && points.at(index).type == 2)
                     || (data.type == 2 && points.at(index).type == 2)){
                    
                    CvPoint p1 = points.at(index).end;
                    CvPoint p2 = data.start;
                    points.at(index).end = cvPoint((p1.x + p2.x)/2, (p1.y + p2.y)/2);
                    data.start = cvPoint((p1.x + p2.x)/2, (p1.y + p2.y)/2);
                }
                // fix & sac
                else if(data.type == 2 && points.at(index).type == 1){
                    
                    CvPoint p1 = points.at(index).start;
                    CvPoint p2 = data.start;
                    points.at(index).start = cvPoint((p1.x + p2.x)/2, (p1.y + p2.y)/2);
                    data.start = cvPoint((p1.x + p2.x)/2, (p1.y + p2.y)/2);
                }
            }
            
            //cout<< fps << " " << data.duration << "ooo" << endl;
            
            
            // add new point
            points.push_back(data);

        }
        
    }
    cout<< "frame time" << frameTime << endl;
    
    //cout<< points.at(0).start.x << " " << points.at(0).start.y << "000000000" << endl;
    
}




int main(){
    
    
    
    
    
    int key = 0;
    
    // Initialize AVI file
    CvCapture* capture = cvCaptureFromAVI( "/Users/lin/Desktop/2903-1.avi" );
    
    IplImage* frame = cvQueryFrame( capture );
    
    
    
    // Check
    if ( !capture ){
        if(file_exists( "/Users/lin/Desktop/2903-2.avi" )) {
            fprintf( stderr, "Cannot open AVI!\n" );
        }else {
            fprintf( stderr, "Cannot find AVI!\n" );
        }
        return 1;
    }
    
    CvVideoWriter* writer = NULL;
    // Get the fps, needed to set the delay
    fps = ( int )cvGetCaptureProperty( capture, CV_CAP_PROP_FPS );
    
    //cout<< "fps " << fps << endl;
    
    //writer = cvCreateVideoWriter("/Users/lin/Desktop/out.avi", CV_FOURCC('W','R','L','E'), fps, size, 1);
    
    // Create a window to display the video
    cvNamedWindow( "video", CV_WINDOW_AUTOSIZE );
    
    // Setup output video
    cv::VideoWriter output_cap;
    
    // read data
    readData();
    
    
    
    
    
    
    bool isFirst = true;
    int count = 0,
        frameIndex = 0;
    
    cout << framePoint.back() << endl;
    /*
    while( count < framePoint.back() && key != 'x' ){
        
        // get the image frame
        frame = cvQueryFrame( capture );
        
        // exit if unsuccessful
        if( !frame ) break;
        
        if(isFirst){
            
            checkFocusRect(points.at(frameIndex));
            
            output_cap.open ( "/Users/lin/Desktop/out.avi", CV_FOURCC('W','R','L','E'), 30, cvSize(resizeWidth, resizeHeight), 1 );
            isFirst = false;
            
            
        }
        
        //frame = transformImg(frame);
        Mat newImg = transformImg(frame, frameIndex);
                
        IplImage iplImage = newImg;
        
        //frame = transformImg(frame, points.at(frameIndex).x, points.at(frameIndex).y);

        //IplImage *iplImage;
        
        //iplImage = cvCreateImage(cvSize(640,480), IPL_DEPTH_8U, 1);
        //iplImage->imageData = (char *) newImg.data;
        
        
        //cvCircle(  &iplImage, targetPoint, 20, cvScalar(255,255,255), -1 );
        
        

        
        // display current frame
        imshow( "video", newImg );
        //cvShowImage("video",&iplImage);
        //cvShowImage("video",&frame);
        //cvWriteFrame(writer, &iplImage);
        //output_cap.write(newImg);
        //cvShowImage("show1",frame);
        
        
        if(count == framePoint.at(frameIndex)){
            frameIndex ++;
            checkFocusRect(points.at(frameIndex));
        }
        count ++;
        // exit if user presses 'x'
        key = cvWaitKey( 1000 / fps );
        
                
    }
     */
    
    while( count < framePoint.back() && key != 'x' ){
        
        // get the image frame
        frame = cvQueryFrame( capture );
        
        // exit if unsuccessful
        if( !frame ) break;
        
        
        if(isFirst){
            //checkFocusRect(points.at(frameIndex));
            imgWidth = cvGetSize(frame).width;
            imgHeight = cvGetSize(frame).height;
            isFirst = false;
        }
         
        
        //frame = transformImg(frame);
        transformImg(frame, frameIndex);
        
                
        imshow( "video", calibratedImg.at(count) );
        
        //cvShowImage("video",&iplImage);
                
        if(count == framePoint.at(frameIndex) || count > framePoint.at(frameIndex) ){
            frameIndex ++;
            //checkFocusRect(points.at(frameIndex));
        }
        count ++;
        cout<< count << endl;
        key = cvWaitKey( 1000 / fps );
    }
        
    outputImg = drawAllData(outputImg);
    cvNamedWindow( "output", CV_WINDOW_AUTOSIZE );
    imshow("output",outputImg);
    
    printf("focus time: %f", focusTime);
    
    cvWaitKey(0);
    // Tidy up
    cvDestroyWindow( "video" );
    cvReleaseCapture( &capture );
    cvReleaseVideoWriter( &writer );
    
    return 0;
}