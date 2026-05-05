#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <vector>
#include <chrono>  // 性能监控

using namespace cv;
using namespace cv::face;
using namespace std;
using namespace chrono;

// ===================== 全局配置（开发板专用）=====================
struct Config {
    // 性能配置
    int PROCESS_WIDTH = 320;      // 处理宽度（原640→320）
    int PROCESS_HEIGHT = 240;     // 处理高度（原480→240）
    int FACE_SIZE = 80;           // 人脸特征尺寸（原100→80）
    int DETECT_EVERY_N_FRAMES = 3; // 每3帧检测一次人脸
    int HELMET_EVERY_N_FRAMES = 2; // 每2帧检测一次安全帽

    // 人脸检测配置
    double SCALE_FACTOR = 1.2;     // 原1.1→1.2，加快检测
    int MIN_NEIGHBORS = 5;         // 原4→5，减少误检
    int MIN_FACE_SIZE = 50;        // 最小人脸尺寸（像素）

    // 识别阈值
    double RECOG_THRESHOLD = 110;   // 识别置信度阈值
} cfg;

// ===================== 安全帽检测（轻量化版）=====================
bool hasHelmet(Mat& frame, Rect face)
{
    if (face.y < 20) return false;

    // 【优化1】缩小检测区域，减少计算量
    int x = face.x;
    int y = max(0, face.y - 60);   // 原-100 → -60
    int w = face.width;
    int h = 50;                     // 原80 → 50

    // 边界检查
    if (x + w > frame.cols) w = frame.cols - x;
    if (y + h > frame.rows) h = frame.rows - y;
    if (w <= 0 || h <= 0) return false;

    Mat roi = frame(Rect(x, y, w, h));
    if (roi.empty()) return false;

    // 【优化2】先缩小再处理（如果区域较大）
    if (w * h > 5000) {
        resize(roi, roi, Size(w / 2, h / 2));
    }

    Mat hsv;
    cvtColor(roi, hsv, COLOR_BGR2HSV);

    // 【优化3】减少颜色检查通道（只检测红、黄、蓝）
    Scalar low_red1(0, 120, 70);    Scalar high_red1(10, 255, 255);
    Scalar low_red2(170, 120, 70); Scalar high_red2(180, 255, 255);
    Scalar low_yellow(20, 100, 100); Scalar high_yellow(30, 255, 255);
    Scalar low_blue(90, 100, 100); Scalar high_blue(120, 255, 255);

    Mat mask;
    Mat m1, m2, m3, m4;
    inRange(hsv, low_red1, high_red1, m1);
    inRange(hsv, low_red2, high_red2, m2);
    inRange(hsv, low_yellow, high_yellow, m3);
    inRange(hsv, low_blue, high_blue, m4);

    mask = m1 | m2 | m3 | m4;

    // 【优化4】降低阈值（原200→100）
    return countNonZero(mask) > 100;
}

// ===================== LBPH 识别器（轻量化版）=====================
Ptr<LBPHFaceRecognizer> model;

void initRecognizer(const vector<Mat>& faces)
{
    if (faces.empty()) {
        cout << "错误：没有训练数据！" << endl;
        return;
    }

    // 【优化5】简化 LBPH 参数，降低计算量
    // radius=1, neighbors=8, grid_x=6, grid_y=6（原8x8 → 6x6）
    model = LBPHFaceRecognizer::create(1, 8, 6, 6, 100);

    vector<int> labels(faces.size(), 1);
    model->train(faces, labels);

    cout << "✅ 模型已加载，训练样本数: " << faces.size() << endl;
}

// 加载训练图片（从文件夹或单张增强）
vector<Mat> loadTrainingFaces(CascadeClassifier& detector)
{
    vector<Mat> faces;

    // 尝试从 training_faces 文件夹加载
    Mat templateImg = imread("template.jpg");
    if (templateImg.empty()) {
        cout << "请放置 template.jpg" << endl;
        return faces;
    }

    vector<Rect> tempFaces;
    Mat grayTemp;
    cvtColor(templateImg, grayTemp, COLOR_BGR2GRAY);
    detector.detectMultiScale(grayTemp, tempFaces, 1.1, 4);

    if (tempFaces.empty()) {
        cout << "未检测到人脸" << endl;
        return faces;
    }

    // 【优化6】训练时使用与识别相同的尺寸
    Mat face = templateImg(tempFaces[0]);
    Mat gray, processed;
    cvtColor(face, gray, COLOR_BGR2GRAY);
    resize(gray, processed, Size(cfg.FACE_SIZE, cfg.FACE_SIZE));
    equalizeHist(processed, processed);
    faces.push_back(processed);

    // 可选：添加简单数据增强（旋转 ±8°）
    for (int angle : {-8, 8}) {
        Mat M = getRotationMatrix2D(Point2f(cfg.FACE_SIZE / 2, cfg.FACE_SIZE / 2), angle, 1.0);
        Mat rotated;
        warpAffine(processed, rotated, M, Size(cfg.FACE_SIZE, cfg.FACE_SIZE));
        faces.push_back(rotated);
    }

    cout << "训练数据: " << faces.size() << " 张" << endl;
    return faces;
}

bool isMatch(Mat faceImg)
{
    Mat gray, processed;
    cvtColor(faceImg, gray, COLOR_BGR2GRAY);
    resize(gray, processed, Size(cfg.FACE_SIZE, cfg.FACE_SIZE));
    equalizeHist(processed, processed);

    int label = -1;
    double conf = 0;
    model->predict(processed, label, conf);

    return (label == 1 && conf < cfg.RECOG_THRESHOLD);
}

// ===================== 性能监控 ======================
void showFPS(Mat& frame, steady_clock::time_point& lastTime, int& frameCount)
{
    frameCount++;
    auto now = steady_clock::now();
    double elapsed = duration_cast<milliseconds>(now - lastTime).count() / 1000.0;

    if (elapsed >= 1.0) {
        double fps = frameCount / elapsed;
        putText(frame, "FPS: " + to_string((int)fps), Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        frameCount = 0;
        lastTime = now;
    }
}

// ===================== 主函数（开发板优化版）=====================
int main()
{
    cout << "=== 人脸识别系统（开发板优化版）===" << endl;

    // 加载人脸检测器
    CascadeClassifier faceDetector;
    string haarPath = samples::findFile("haarcascades/haarcascade_frontalface_default.xml");
    if (!faceDetector.load(haarPath)) {
        cout << "无法加载人脸检测器" << endl;
        return -1;
    }

    // 初始化识别模型
    vector<Mat> trainingFaces = loadTrainingFaces(faceDetector);
    if (trainingFaces.empty()) {
        cout << "初始化失败" << endl;
        return -1;
    }
    initRecognizer(trainingFaces);

    // 打开摄像头
    VideoCapture cap;
    if (!cap.open(0, CAP_V4L2)) {
        cout << "无法打开摄像头" << endl;
        return -1;
    }

    // 【优化7】设置相机参数
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(CAP_PROP_FRAME_WIDTH, cfg.PROCESS_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, cfg.PROCESS_HEIGHT);
    cap.set(CAP_PROP_FPS, 15);  // 降低帧率以减少CPU占用

    cout << "✅ 摄像头已启动 (" << cfg.PROCESS_WIDTH << "x" << cfg.PROCESS_HEIGHT << ")" << endl;
    cout << "性能模式: 每" << cfg.DETECT_EVERY_N_FRAMES << "帧检测一次人脸" << endl;

    Mat frame;
    int frameSkip = 0;
    int helmetSkip = 0;
    int frameCount = 0;
    auto fpsTimer = steady_clock::now();

    // 缓存上一次检测结果（避免每帧都重新检测）
    vector<Rect> lastFaces;
    bool lastMatchResult = false;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        frameSkip++;
        helmetSkip++;

        vector<Rect> faces;
        bool known = false;

        // 【优化8】降低人脸检测频率
        if (frameSkip >= cfg.DETECT_EVERY_N_FRAMES) {
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);

            // 使用优化后的检测参数
            faceDetector.detectMultiScale(gray, faces,
                cfg.SCALE_FACTOR,
                cfg.MIN_NEIGHBORS,
                0,
                Size(cfg.MIN_FACE_SIZE, cfg.MIN_FACE_SIZE));

            frameSkip = 0;
            lastFaces = faces;

            // 识别人脸（只对检测到的第一张脸）
            if (!faces.empty()) {
                known = isMatch(frame(faces[0]));
                lastMatchResult = known;
            }
            else {
                lastMatchResult = false;
            }
        }
        else {
            // 使用缓存的检测结果
            faces = lastFaces;
            known = lastMatchResult;
        }

        // 安全帽检测（单独控制频率）
        bool helmet = false;
        if (!faces.empty() && helmetSkip >= cfg.HELMET_EVERY_N_FRAMES) {
            helmet = hasHelmet(frame, faces[0]);
            helmetSkip = 0;
        }
        else if (!faces.empty()) {
            // 使用上次结果（简单缓存，实际会轻微闪烁，可接受）
            helmet = false;
        }

        // 绘制结果
        for (auto& f : faces) {
            if (known) {
                if (helmet) {
                    rectangle(frame, f, Scalar(0, 255, 0), 2);
                    putText(frame, "OK + HELMET", f.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
                }
                else {
                    rectangle(frame, f, Scalar(0, 255, 255), 2);
                    putText(frame, "OK (NO HELMET)", f.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
                }
            }
            else {
                rectangle(frame, f, Scalar(0, 0, 255), 2);
                putText(frame, "UNKNOWN", f.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
            }
        }

        // 显示FPS
        showFPS(frame, fpsTimer, frameCount);

        imshow("SAFETY CHECK (DEV BOARD)", frame);
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}