#include"testRanger.h"
#include <ranger/ArgumentHandler.h>
#include <ranger/DataFloat.h>
#include <ranger/ForestClassification.h>
#include <ranger/ForestProbability.h>

#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>

void StringAppendV(std::string *dst, const char *format, va_list ap) {
	// First try with a small fixed size buffer.
	static const int kFixedBufferSize = 1024;
	char fixed_buffer[kFixedBufferSize];

	// It is possible for methods that use a va_list to invalidate
	// the data in it upon use.  The fix is to make a copy
	// of the structure before using it and use that copy instead.
	va_list backup_ap;
	va_copy(backup_ap, ap);
	int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
	va_end(backup_ap);

	if (result < kFixedBufferSize) {
		if (result >= 0) {
			// Normal case - everything fits.
			dst->append(fixed_buffer, result);
			return;
		}

#ifdef _MSC_VER
		// Error or MSVC running out of space.  MSVC 8.0 and higher
		// can be asked about space needed with the special idiom below:
		va_copy(backup_ap, ap);
		result = vsnprintf(nullptr, 0, format, backup_ap);
		va_end(backup_ap);
#endif

		if (result < 0) {
			// Just an error.
			return;
		}
	}

	// Increase the buffer size to the size requested by vsnprintf,
	// plus one for the closing \0.
	const int variable_buffer_size = result + 1;
	std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

	// Restore the va_list before we use it again.
	va_copy(backup_ap, ap);
	result = vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
	va_end(backup_ap);

	if (result >= 0 && result < variable_buffer_size) {
		dst->append(variable_buffer.get(), result);
	}
}
std::string string_printf(const char *format, ...) {
	va_list ap;
	va_start(ap, format);
	std::string result;
	StringAppendV(&result, format, ap);
	va_end(ap);
	return result;
}

/************************************************************************/
/*                     SuperpixelClassifier                             */
/************************************************************************/
SuperpixelClassifier::SuperpixelClassifier() { rf_ = std::make_shared<ForestClassification>(); }
SuperpixelClassifier::~SuperpixelClassifier() {}

void SuperpixelClassifier::train(const MatrixXf &features, const VectorXf &labels) {
	ArgumentHandler args(0, NULL);
	int num_data = features.rows();
	int num_feat = features.cols();
	MatrixXf combined = MatrixXf::Zero(num_data, num_feat + 1);
	combined << features, labels;

	std::vector<std::string> variable_names;
	for (int i = 0; i < num_feat; i++) {
		std::string name = string_printf("feat #%d", i);
		variable_names.push_back(name);
	}
	variable_names.push_back("label");

	try {
		// data 和 eigen 都采用的是 col major 存储
		std::unique_ptr<DataFloat> data =
			std::make_unique<DataFloat>(combined.data(), variable_names, combined.rows(), combined.cols());

		rf_->init("label", MEM_FLOAT, data.get(), args.mtry, args.outprefix, args.ntree, args.seed, args.nthreads,
			args.impmeasure, args.targetpartitionsize, args.statusvarname, false, args.replace, args.catvars,
			args.savemem, args.splitrule, args.predall, args.fraction, args.alpha, args.minprop, args.holdout,
			args.predictiontype, args.randomsplits);

		rf_->run(false);

	}
	catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << " Ranger will EXIT now." << std::endl;
		return;
	}
}
VectorXf SuperpixelClassifier::predict(const MatrixXf &features) {

	ArgumentHandler args(0, NULL);
	int num_data = features.rows();
	int num_feat = features.cols();

	std::vector<std::string> variable_names;
	for (int i = 0; i < num_feat; i++) {
		std::string name = string_printf("feat #%d", i);
		variable_names.push_back(name);
	}
	VectorXf labels(num_data);
	try {
		// data 内部用的是 row major，但是 eigen 采用的是 col major，所以这里只要颠倒行列的意义即可
		DataFloat *data = new DataFloat(features.data(), variable_names, features.rows(), features.cols());

		rf_->init("label", MEM_FLOAT, data, args.mtry, args.outprefix, args.ntree, args.seed, args.nthreads,
			args.impmeasure, args.targetpartitionsize, args.statusvarname, true, args.replace, args.catvars,
			args.savemem, args.splitrule, args.predall, args.fraction, args.alpha, args.minprop, args.holdout,
			args.predictiontype, args.randomsplits);
		std::vector<std::string> empty;
		data->setIsOrderedVariable(empty);
		rf_->run(false);

		std::vector<double> result = rf_->getPredictions()[0][0];
		for (int i = 0; i < result.size(); ++i) {
			labels(i) = result[i];
		}
		delete data;
	}
	catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << " Ranger will EXIT now." << std::endl;
		return VectorXf();
	}

	return labels;
}
void SuperpixelClassifier::save(const std::string &path) {}
void SuperpixelClassifier::load(const std::string &path) {}
/************************************************************************/
/*                    SuperpixelClassifierProbability                   */
/************************************************************************/
SuperpixelClassifierProbability::SuperpixelClassifierProbability() { rf_ = std::make_shared<ForestProbability>(); }
SuperpixelClassifierProbability::~SuperpixelClassifierProbability() {}
void SuperpixelClassifierProbability::train(const MatrixXf &features, const VectorXf &labels) {
	ArgumentHandler args(0, NULL);
	int num_data = features.rows();
	int num_feat = features.cols();
	MatrixXf combined = MatrixXf::Zero(num_data, num_feat + 1);
	combined << features, labels;

	std::vector<std::string> variable_names;
	for (int i = 0; i < num_feat; i++) {
		std::string name = string_printf("feat #%d", i);
		variable_names.push_back(name);
	}
	variable_names.push_back("label");

	try {
		// data 和 eigen 都采用的是 col major 存储
		DataFloat *data = new DataFloat(combined.data(), variable_names, combined.rows(), combined.cols());

		rf_->init("label", MEM_FLOAT, data, args.mtry, args.outprefix, args.ntree, args.seed, args.nthreads,
			args.impmeasure, args.targetpartitionsize, args.statusvarname, false, args.replace, args.catvars,
			args.savemem, args.splitrule, args.predall, args.fraction, args.alpha, args.minprop, args.holdout,
			args.predictiontype, args.randomsplits);

		rf_->run(false);

		delete data;
	}
	catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << " Ranger will EXIT now." << std::endl;
		return;
	}
}
MatrixXf SuperpixelClassifierProbability::predict(const MatrixXf &features) {
	ArgumentHandler args(0, NULL);
	int num_data = features.rows();
	int num_feat = features.cols();

	std::vector<std::string> variable_names;
	for (int i = 0; i < num_feat; i++) {
		std::string name = string_printf("feat #%d", i);
		variable_names.push_back(name);
	}
	MatrixXf probabilities;
	try {
		// data 内部用的是 row major，但是 eigen 采用的是 col major，所以这里只要颠倒行列的意义即可
		DataFloat *data = new DataFloat(features.data(), variable_names, features.rows(), features.cols());

		rf_->init("label", MEM_FLOAT, data, args.mtry, args.outprefix, args.ntree, args.seed, args.nthreads,
			args.impmeasure, args.targetpartitionsize, args.statusvarname, true, args.replace, args.catvars,
			args.savemem, args.splitrule, args.predall, args.fraction, args.alpha, args.minprop, args.holdout,
			args.predictiontype, args.randomsplits);
		std::vector<std::string> empty;
		data->setIsOrderedVariable(empty);
		rf_->run(false);
		std::vector<double> class_values = ((ForestProbability *)(rf_.get()))->getClassValues();
		probabilities = MatrixXf::Zero(num_data, class_values.size());
		auto results = rf_->getPredictions()[0];
		for (int k = 0; k < class_values.size(); ++k) {
			for (int i = 0; i < num_data; ++i) {
				int classid = int(class_values[k] + 0.5);
				probabilities(i, k) = results[i][classid];
			}
		}

		delete data;
	}
	catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << " Ranger will EXIT now." << std::endl;
		return MatrixXf();
	}

	return probabilities;
}
void SuperpixelClassifierProbability::save(const std::string &path) {}
void SuperpixelClassifierProbability::load(const std::string &path) {}