#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>

#define QSB_SE_PER_EPOCH 256
#define QSB_SE_TWIN 3
#define QSB_SE_CUT 137
#define SIG_PUSH_SIZE 10
#define QSB_SCHEDULE_SELECT_ONLY 1
#include "gpu_epochs/window_schedule_shared.cuh"

int main() {
    uint8_t rows[150 * SIG_PUSH_SIZE];
    for (int i = 0; i < 150; ++i)
        for (int j = 0; j < SIG_PUSH_SIZE; ++j)
            rows[i * SIG_PUSH_SIZE + j] =
                static_cast<uint8_t>(i * 31 + j * 17 + i * j * 3);

    uint8_t windows[QSB_SE_PER_EPOCH][QSB_SE_TWIN];
    assert(qsb_select_window_schedule(rows, windows) == 0);

    std::set<std::array<int, 3>> triples;
    std::set<std::array<uint8_t, 56>> first_classes;
    std::set<std::array<uint8_t, 44>> second_classes;
    std::array<uint8_t, 44> previous_second{};
    bool have_previous = false;
    int second_runs = 0;
    for (int lane = 0; lane < QSB_SE_PER_EPOCH; ++lane) {
        std::array<int, 3> triple{};
        for (int j = 0; j < 3; ++j) triple[j] = windows[lane][j] - QSB_SE_CUT;
        assert(0 <= triple[0] && triple[0] < triple[1] &&
               triple[1] < triple[2] && triple[2] < 13);
        triples.insert(triple);

        uint8_t kept[100];
        int pos = 0;
        for (int i = 0; i < 13; ++i) {
            if (i == triple[0] || i == triple[1] || i == triple[2]) continue;
            memcpy(kept + pos, rows + (QSB_SE_CUT + i) * SIG_PUSH_SIZE,
                   SIG_PUSH_SIZE);
            pos += SIG_PUSH_SIZE;
        }
        assert(pos == 100);
        std::array<uint8_t, 56> first{};
        std::array<uint8_t, 44> second{};
        memcpy(first.data(), kept, first.size());
        memcpy(second.data(), kept + first.size(), second.size());
        first_classes.insert(first);
        second_classes.insert(second);
        if (!have_previous || second != previous_second) {
            ++second_runs;
            previous_second = second;
            have_previous = true;
        }
    }

    assert(triples.size() == QSB_SE_PER_EPOCH);
    assert(first_classes.size() == 54);
    assert(second_classes.size() == 56);
    assert(second_runs == 56);
    std::printf("host selector: lanes=%zu first=%zu second=%zu runs=%d\n",
                triples.size(), first_classes.size(), second_classes.size(), second_runs);
    return 0;
}
