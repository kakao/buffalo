#pragma once
#include <omp.h>
#include <vector>
#include <string>
#include <cstdio>
#include <fstream>

using namespace std;

namespace fileio
{

vector<string> _chunking_file(string path, string to_dir, long total_lines, int num_chunks, int sep_idx, int num_workers)
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

}
