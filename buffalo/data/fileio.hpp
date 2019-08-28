#pragma once
#include <cmath>
#include <omp.h>
#include <vector>
#include <string>
#include <cstdio>
#include <fstream>
#include <parallel/algorithm>

using namespace std;

namespace fileio
{

vector<string> _chunking_into_bins(string path, string to_dir,
        int64_t total_lines, int num_chunks, int sep_idx, int num_workers)
{
    ifstream fin(path.c_str());
    if (!fin.is_open()) {
        return {};
    }

    long per_chunk_lines = total_lines / num_chunks + 1;

    char cbuffer[512];
    FILE* fouts[num_chunks];
    for (int i=0; i < num_chunks; ++i) {
        sprintf(cbuffer, "%s/chunk%d.txt", to_dir.c_str(), i);
        fouts[i] = fopen(cbuffer, "w");
    }

    string line;
    int fout_id = 0;
    long chunk_size = 0;
    int prev_id = -1, cur_id = 0;
    while (getline(fin, line)) {
        if (chunk_size > per_chunk_lines) {
            if (sep_idx == 0)
                sscanf(line.c_str(), "%d %*d %*d", &cur_id);
            else
                sscanf(line.c_str(), "%*d %d %*d", &cur_id);

            if (prev_id != -1 and prev_id != cur_id) {
                fout_id += 1;
                assert(fout_id < num_chunks);
                chunk_size = 0;
                prev_id = -1;
            } else {
                prev_id = cur_id;
            }
            fprintf(fouts[fout_id], "%s\n", line.c_str());
        }
        else {
            fprintf(fouts[fout_id], "%s\n", line.c_str());
        }
        chunk_size += 1;
    }
    fin.close();

    vector<string> tmp_files;
    vector<string> chunk_files;
    ifstream fins[num_chunks];
    for (int i=0; i < num_chunks; ++i) {
        fclose(fouts[i]);
        sprintf(cbuffer, "%s/chunk%d.txt", to_dir.c_str(), i);
        tmp_files.push_back(cbuffer);
        fins[i].open(cbuffer);
        sprintf(cbuffer, "%s/chunk%d.bin", to_dir.c_str(), i);
        fouts[i] = fopen(cbuffer, "w");
        chunk_files.push_back(cbuffer);
    }

    omp_set_num_threads(num_workers);
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1)
        for (int i=0; i < num_chunks; ++i) {
            string _line;
            int r, c;
            float v;
            while (getline(fins[i], _line)) {
                sscanf(_line.c_str(), "%d %d %f", &r, &c, &v);
                fwrite((char*)&r, sizeof(r), 1, fouts[i]);
                fwrite((char*)&c, sizeof(c), 1, fouts[i]);
                fwrite((char*)&v, sizeof(v), 1, fouts[i]);
            }
        }
    }
    for (int i=0; i < num_chunks; ++i) {
        fins[i].close();
        unlink(tmp_files[i].c_str());
        fclose(fouts[i]);
    }
    return chunk_files;
}


struct triple_t
{
    int r, c;
    float v;
    triple_t() : r(0), c(0), v(0.0) {
    }
};

string _sort_and_compressed_binarization(
        string path,
        string to_dir,
        int64_t total_lines,
        int max_key,
        int sort_key,
        int num_workers)
{
    FILE* fin = fopen(path.c_str(), "r");

    vector<triple_t> records(total_lines);
    for (long i=0; i < total_lines; ++i) {
        fscanf(fin, "%d %d %f", &records[i].r, &records[i].c, &records[i].v);
    }

    omp_set_num_threads(num_workers);

    if (sort_key > 0) {
        __gnu_parallel::stable_sort(records.begin(),
                                    records.end(),
                                    [sort_key](const triple_t& t1, const triple_t& t2){
                                        if (sort_key == 1) {
                                            if (t1.r == t2.r) return t1.c < t2.c;
                                            else return t1.r < t2.r;
                                        } else {
                                            if (t1.c == t2.c) return t1.r < t2.r;
                                            else return t1.c < t2.c;
                                        }
                                    });
    }

    char cbuffer[512];
    sprintf(cbuffer, "%s/chunk.bin", to_dir.c_str());
    FILE* fout = fopen(cbuffer, "w");

    vector<int64_t> indptr;
    indptr.reserve(max_key);

    if (sort_key == 1 or sort_key == -1) {
        for (int j=0; j<records[0].r-1; ++j)
            indptr.push_back(0);
        for (int64_t i=1; i < total_lines; ++i) {
            for (int j=0; j < records[i].r - records[i - 1].r; ++j)
                indptr.push_back(i);
        }
        for (int j=0; j<max_key+1-records[total_lines-1].r; ++j)
            indptr.push_back(total_lines);
    }
    else {
        for (int j=0; j<records[0].c-1; ++j)
            indptr.push_back(0);
        for (int64_t i=1; i < total_lines; ++i) {
            for (int j=0; j < records[i].c - records[i - 1].c; ++j)
                indptr.push_back(i);
        }
        for (int j=0; j<max_key+1-records[total_lines-1].c; ++j)
            indptr.push_back(total_lines);
    }

    for (const auto& i : indptr) {
        fwrite((char*)&i, sizeof(i), 1, fout);
    }

    if (sort_key == 1 or sort_key == -1){
        for (const auto& r : records) {
            fwrite((char*)&r.c, sizeof(r.c), 1, fout);
            fwrite((char*)&r.v, sizeof(r.v), 1, fout);
        }
    }
    else {
        for (const auto& r : records) {
            fwrite((char*)&r.r, sizeof(r.r), 1, fout);
            fwrite((char*)&r.v, sizeof(r.v), 1, fout);
        }
    }

    fclose(fout);
    return string(cbuffer);
}

}
