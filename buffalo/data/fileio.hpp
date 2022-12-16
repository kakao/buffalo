#pragma once
#include <cmath>
#include <omp.h>
#include <vector>
#include <unordered_set>
#include <string>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <atomic>
#include <iostream>
#include <parallel/algorithm>

using namespace std;

namespace fileio
{

ifstream::pos_type get_file_size(const char* fname)
{
    ifstream in(fname, ios::ate);
    return in.tellg();
}

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
                --r;
                --c;
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

int _parallel_build_sppmi(string from_path, string to_path,
        int64_t total_lines, int num_items, int k, int num_workers)
{
    double log_d = log(total_lines);
    double log_k = log(k);

    atomic<int64_t> appearances[num_items];
    for (int i=0; i < num_items; ++i) {
        appearances[i] = 0;
    }

    ofstream fout(to_path.c_str(), ofstream::out | ofstream::trunc);
    int64_t nnz = 0;

    long file_size = get_file_size(from_path.c_str());

    long split_size = 4 * 1024 * 1024;
    long num_split = file_size / split_size;
    if (file_size % split_size != 0) {
        ++num_split;
    }

    #pragma omp parallel num_threads(num_workers)
    {
        ifstream fin(from_path.c_str());

        // build appearances
        #pragma omp for schedule(dynamic, 1)
        for (int i=0; i < num_split; ++i) {
            long start_pos = i*split_size;
            long end_pos = (i+1) * split_size;

            fin.clear();
            fin.seekg(start_pos, ios::beg);

            string line;
            if (i != 0) {
                // first line is handled by previous thread
                getline(fin, line);
            }

            while(getline(fin, line)) {
                int cur_id;
                sscanf(line.c_str(), "%d %*d", &cur_id);
                assert(cur_id > 0 and cur_id <= num_items);

                appearances[cur_id-1].fetch_add(1, memory_order_relaxed);

                // not less than. for processing first line of next split
                if (end_pos < fin.tellg()) {
                    break;
                }
            }
        }

        // build sppmi
        #pragma omp for schedule(dynamic, 1)
        for (int i=0; i < num_split; ++i) {
            long start_pos = i*split_size;
            long end_pos = (i+1) * split_size;

            fin.clear();
            fin.seekg(start_pos, ios::beg);

            string line;
            if (i != 0) {
                // first line is handled by previous thread
                getline(fin, line);
            }

            int skip_id = -1;
            int last_id = -1;
            int probe_id = -1;
            bool break_if_new_id_begin = false;
            vector<int> chunk;
            while(getline(fin, line)) {
                int cur_id, c;
                sscanf(line.c_str(), "%d %d", &cur_id, &c);
                assert(cur_id > 0 and cur_id <= num_items);

                // first id found is handled by previous thread
                if (i != 0 and (skip_id == -1 or skip_id == cur_id)) {
                    skip_id = cur_id;

                    if (break_if_new_id_begin) {
                        break;
                    }

                    // not less than. for processing first line of next split
                    if (end_pos < fin.tellg()) {
                        break_if_new_id_begin = true;
                    }

                    continue;
                }

                if (probe_id == -1) {
                    probe_id = cur_id;
                } else if (probe_id != cur_id) {
                    unordered_set<int> chunk_set(chunk.begin(), chunk.end());
                    for (const auto& _c : chunk_set) {
                        if (probe_id < _c)
                            continue;
                        int cnt = count(chunk.begin(), chunk.end(), _c);
                        double pmi = log(cnt) + log_d
                            - log(appearances[probe_id-1].load(memory_order_relaxed))
                            - log(appearances[_c-1].load(memory_order_relaxed));
                        double sppmi = pmi - log_k;

                        if (sppmi > 0) {
                            #pragma omp critical(out)
                            {
                                fout << probe_id << ' ' << _c << ' ' << sppmi << '\n';
                                fout << _c << ' ' << probe_id << ' ' << sppmi << '\n';
                                nnz += 2;
                            }
                        }
                    }

                    probe_id = cur_id;
                    chunk.clear();
                }
                chunk.push_back(c);

                if (break_if_new_id_begin) {
                    if (last_id == -1) {
                        last_id = cur_id;
                    } else if (last_id != cur_id) {
                        break;
                    }
                }

                // not less than. for processing first line of next split
                if (end_pos < fin.tellg()) {
                    break_if_new_id_begin = true;
                }
            }
        }
    }

    return nnz;
}


struct triple_t
{
    int r, c;
    float v;
    triple_t() : r(0), c(0), v(0.0) {
    }
    triple_t(int r, int c, float v) : r(r), c(c), v(v) {
    }
};

vector<string> _sort_and_compressed_binarization(
        string path,
        string to_dir,
        int64_t total_lines,
        int max_key,
        int sort_key,
        int num_workers)
{
    long file_size = get_file_size(path.c_str());

    long split_size = 4 * 1024 * 1024;
    long num_split = file_size / split_size;
    if (file_size % split_size != 0) {
        ++num_split;
    }

    vector<vector<triple_t>> split_records(num_split);

    #pragma omp parallel num_threads(num_workers)
    {
        ifstream fin(path.c_str());

        #pragma omp for schedule(dynamic, 1)
        for (int i=0; i < num_split; ++i) {
            long start_pos = i*split_size;
            long end_pos = (i+1) * split_size;

            fin.clear();
            fin.seekg(start_pos, ios::beg);

            vector<triple_t> records;
            string line;
            if (i != 0) {
                // first line is handled by previous thread
                getline(fin, line);
            }

            while(getline(fin, line)) {
                int r, c;
                float v;
                sscanf(line.c_str(), "%d %d %f", &r, &c, &v);
                records.emplace_back(r, c, v);

                // not less than. for processing first line of next split
                if (end_pos < fin.tellg()) {
                    break;
                }
            }
            split_records[i] = records;
        }
    }

    vector<triple_t> records;
    records.reserve(total_lines);
    for (const auto& v : split_records) {
        auto end_it = end(v);
        if (records.size() + v.size() > (uint64_t)total_lines) {
            end_it = next(begin(v), total_lines - records.size());
        }
        records.insert(end(records), begin(v), end_it);
    }

    assert(records.size == total_lines);

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

    vector<string> chunk_files;
    char cbuffer[512];
    sprintf(cbuffer, "%s/indptr.bin", to_dir.c_str());
    FILE* fout_indptr = fopen(cbuffer, "w");
    chunk_files.push_back(cbuffer);

    FILE* fouts[num_workers];
    for (int i=0; i < num_workers; ++i) {
        sprintf(cbuffer, "%s/chunk%d.bin", to_dir.c_str(), i);
        fouts[i] = fopen(cbuffer, "w");
        chunk_files.push_back(cbuffer);
    }

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
        fwrite((char*)&i, sizeof(i), 1, fout_indptr);
    }

    long records_size = records.size();
    long per_workers_num = records_size / num_workers;
    if (records_size % num_workers != 0) {
        ++per_workers_num;
    }

    #pragma omp parallel num_threads(num_workers)
    {
        #pragma omp for
        for (int i=0; i < num_workers; ++i) {
            long start_index = per_workers_num * i;
            long end_index = min(per_workers_num * (i+1), records_size);

            for (long j=start_index; j < end_index; ++j) {
                auto& r = records[j];
                if (sort_key == 1 or sort_key == -1) {
                    --r.c;
                    fwrite((char*)&r.c, sizeof(r.c), 1, fouts[i]);
                    fwrite((char*)&r.v, sizeof(r.v), 1, fouts[i]);
                }
                else {
                    --r.r;
                    fwrite((char*)&r.r, sizeof(r.r), 1, fouts[i]);
                    fwrite((char*)&r.v, sizeof(r.v), 1, fouts[i]);
                }
            }
        }
    }

    fclose(fout_indptr);
    for (int i=0; i < num_workers; ++i) {
        fclose(fouts[i]);
    }

    return chunk_files;
}

}
