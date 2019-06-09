#include <string>
#include <fstream>
#include <streambuf>

#include "buffalo/algo.hpp"

using namespace std;
using namespace json11;

Algorithm::Algorithm()
{
    logger_ = BuffaloLogger().get_logger();
}

bool Algorithm::parse_option(string opt_path, Json& j)
{
    ifstream in(opt_path.c_str());
    if (not in.is_open()) {
        INFO("File not exists: {}", opt_path);
        return false;
    }

    string str((std::istreambuf_iterator<char>(in)),
               std::istreambuf_iterator<char>());
    string err_cmt;
    auto _j = Json::parse(str, err_cmt);
    if (not err_cmt.empty()) {
        INFO("Failed to parse: {}", err_cmt);
        return false;
    }
    j = _j;
    return true;
}

void Algorithm::decouple(Map<MatrixXf>& mat, float** data, int& rows, int& cols) {
    (*data) = mat.data();
    rows = mat.rows();
    cols = mat.cols();
}
