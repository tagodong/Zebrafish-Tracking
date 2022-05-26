
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <sstream>
#include <algorithm>

using namespace std;

template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}

int cmp(const void* a, const void* b)
{
	const int sa = ((float*)a)[0];
	std::string s1 = to_string(sa);

	const int sb = ((float*)b)[0];
	std::string s2 = to_string(sb);

	return strcmp(s1.c_str(), s2.c_str());
}


void Zebrafish_OD_TRT()
{

	int max_batch_size = 1;
	/** 模型编译，onnx到trtmodel **/

    TRT::compile(
        TRT::Mode::FP16,            /** 模式, fp32 fp16 int8  **/
        max_batch_size,             /** 最大batch size        **/
        "..\\model\\best_OD_weight.onnx",             /** onnx文件，输入         **/
        "..\\model\\best_OD_weight.trtmodel"    /** trt模型文件，输出      **/
    );

	/** 加载编译好的引擎 **/
	auto infer = TRT::load_infer("..\\model\\best_OD_weight.trtmodel");
	// to infer the image
	auto files = iLogger::find_files("..\\..\\test", "*.png");

	auto labels = iLogger::load_text_file("..\\..\\test\\labels.txt");
	auto label_vector = iLogger::split_string(labels, "\n");
	string savepath = "..\\..\\predict";

	float label[7409][5];

	for (size_t i = 0; i < label_vector.size(); i++)
	{
		auto label_temp = iLogger::split_string(label_vector[i], ",");
		for (size_t j = 0; j < label_temp.size() - 1; j++)
		{
			label[i][j] = stringToNum<float>(label_temp[j]);
		}
	}

	qsort(label, 7409, sizeof(int) * 5, cmp);
	double sum_error[4] = { 0,0,0,0 };



	for (int i = 0; i < files.size(); ++i) {

		// cout << "sample name: " << files[i] << endl;
		auto image = cv::imread(files[i]);
		// image preprocess
		float mean[] = { 0.5, 0.5, 0.5 };
		float std[] = { 0.5, 0.5, 0.5 };
		auto t0 = iLogger::timestamp_now_float();
		/** 设置输入的值 **/
		/** 修改input的0维度为1，最大可以是5 **/
		// Yolo::image_to_tensor(image, image_tensor, Yolo::Type::V5, 1);
		// auto t = image_tensor.get();

		infer->input(0)->set_norm_mat(0, image, mean, std);
		/** 引擎进行推理 **/
		infer->forward();
		/** 取出引擎的输出并打印 **/
		auto out = infer->output(0);
		auto t1 = iLogger::timestamp_now_float();
		cout << "time comsumed:" << t1 - t0 << endl;

		cout << "original coordinates: " << label[i][1] * 320.0 / 360.0 << "\t" << label[i][2] * 320.0 / 360.0 << "\t" << label[i][3] * 320.0 / 360.0 << "\t" << label[i][4] * 320.0 / 360.0 << endl;
		cout << "predicted coordinates: " << out->at<float>(0, 0) << "\t" << out->at<float>(0, 1) << "\t" << out->at<float>(0, 2) << "\t" << out->at<float>(0, 3) << endl;


		sum_error[0] += abs(out->at<float>(0, 0) - label[i][1] * 320.0 / 360.0);
		sum_error[1] += abs(out->at<float>(0, 1) - label[i][2] * 320.0 / 360.0);
		sum_error[2] += abs(out->at<float>(0, 2) - label[i][3] * 320.0 / 360.0);
		sum_error[3] += abs(out->at<float>(0, 3) - label[i][4] * 320.0 / 360.0);

		// if show the picture
		//INFO("out.shape = %s", out->shape_string());
		cv::circle(image, cv::Point(round(out->at<float>(0, 0) * 360 / 320), round(out->at<float>(0, 1) * 360 / 320)), 2, (255, 0, 0), 1);
		cv::circle(image, cv::Point(round(out->at<float>(0, 2) * 360 / 320), round(out->at<float>(0, 3) * 360 / 320)), 2, (0, 0, 255), 1);
		//cv::imshow(files[i], image);
		//cv::waitKey();
		/*auto image_name = iLogger::file_name(files[i], true);
		auto save_name = savepath + "\\" + image_name;
		cv::imwrite(save_name, image);
		cout << save_name<< "true" << endl;*/

	}
	double mean_error[4];
	mean_error[0] = sum_error[0] / 7409.0;
	mean_error[1] = sum_error[1] / 7409.0;
	mean_error[2] = sum_error[2] / 7409.0;
	mean_error[3] = sum_error[3] / 7409.0;
	cout << mean_error[0] << endl;
	cout << mean_error[1] << endl;
	cout << mean_error[2] << endl;
	cout << mean_error[3] << endl;
}
int Zebrafish_OD() {

	Zebrafish_OD_TRT();
	return 0;
}