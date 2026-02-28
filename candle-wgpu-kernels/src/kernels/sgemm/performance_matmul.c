2048x2048 * 2048x2048:

16x16:
#define TSM 16u                     // The tile-size in dimension M
#define TSN 16u                     // The tile-size in dimension N
#define TSK 4u                     // The tile-size in dimension K
#define WPTM 4u                     // The work-per-thread in dimension M
#define WPTN 2u                     // The work-per-thread in dimension N
#define WIDTHA 2u
#define WIDTHB 2u
//#define WONT_USE_LOADB
1238.7 GiB/s


16x32:

1721.0 GiB/s
#define TSM 16u                     // The tile-size in dimension M
#define TSN 32u                     // The tile-size in dimension N
#define TSK 4u                     // The tile-size in dimension K
#define WPTM 4u                     // The work-per-thread in dimension M
#define WPTN 4u                     // The work-per-thread in dimension N
#define WIDTHA 2u
#define WONT_USE_LOADB

32x32:
            
2461.1 GiB/s
#define TSM 32u                     // The tile-size in dimension M
#define TSN 32u                     // The tile-size in dimension N
#define TSK 4u                     // The tile-size in dimension K
#define WPTM 8u                     // The work-per-thread in dimension M
#define WPTN 4u                     // The work-per-thread in dimension N

#define WIDTHA 4u
//#define WIDTHB 2u
//#define WONT_USE_LOADB

                2050.3 GiB/s
                #define TSM 32u                     // The tile-size in dimension M
                #define TSN 32u                     // The tile-size in dimension N
                #define TSK 4u                     // The tile-size in dimension K
                #define WPTM 8u                     // The work-per-thread in dimension M
                #define WPTN 4u                     // The work-per-thread in dimension N
                #define WIDTHA 2u
                //#define WIDTHB 2u
                #define WONT_USE_LOADB


                2123.2 GiB/s
                #define TSM 32u                     // The tile-size in dimension M
                #define TSN 32u                     // The tile-size in dimension N
                #define TSK 4u                     // The tile-size in dimension K
                #define WPTM 32u                     // The work-per-thread in dimension M
                #define WPTN 1u                     // The work-per-thread in dimension N
                #define WIDTHA 4u
                #define WIDTHB 1u


                2266.1 GiB/s
                #define TSM 32u                     // The tile-size in dimension M
                #define TSN 32u                     // The tile-size in dimension N
                #define TSK 4u                     // The tile-size in dimension K
                #define WPTM 8u                     // The work-per-thread in dimension M
                #define WPTN 4u                     // The work-per-thread in dimension N
                #define WIDTHA 2u
                //#define WIDTHB 2u
                //#define WONT_USE_LOADB

                2305.7 GiB/s
                #define TSM 32u                     // The tile-size in dimension M
                #define TSN 32u                     // The tile-size in dimension N
                #define TSK 4u                     // The tile-size in dimension K
                #define WPTM 32u                     // The work-per-thread in dimension M
                #define WPTN 1u                     // The work-per-thread in dimension N
                #define WIDTHA 4u
                #define WIDTHB 1u
                #define WONT_USE_LOADB



32x64:

                2517.4 GiB/s
                #define TSM 32u                     // The tile-size in dimension M
                #define TSN 64u                     // The tile-size in dimension N
                #define TSK 4u                     // The tile-size in dimension K
                #define WPTM 16u                     // The work-per-thread in dimension M
                #define WPTN 4u                     // The work-per-thread in dimension N
                #define WIDTHA 4u
                //#define WIDTHB 2u
                //#define WONT_USE_LOADB

                2980.2 Gib/s
                #define TSM 32u                     // The tile-size in dimension M
                #define TSN 64u                     // The tile-size in dimension N
                #define TSK 4u                     // The tile-size in dimension K
                #define WPTM 32u                     // The work-per-thread in dimension M
                #define WPTN 2u                     // The work-per-thread in dimension N
                #define WIDTHA 4u
                #define WIDTHB 2u
                #define WONT_USE_LOADB

                2743.9 GiB/s
                #define TSM 32u                     // The tile-size in dimension M
                #define TSN 64u                     // The tile-size in dimension N
                #define TSK 2u                     // The tile-size in dimension K
                #define WPTM 32u                     // The work-per-thread in dimension M
                #define WPTN 2u                     // The work-per-thread in dimension N
                #define WIDTHA 1u
                #define WIDTHB 1u
                //#define WONT_USE_LOADB

                2877.6 GiB/s
                #define TSM 32u                     // The tile-size in dimension M
                #define TSN 64u                     // The tile-size in dimension N
                #define TSK 4u                     // The tile-size in dimension K
                #define WPTM 32u                     // The work-per-thread in dimension M
                #define WPTN 2u                     // The work-per-thread in dimension N
                #define WIDTHA 1u
                #define WIDTHB 1u
                //#define WONT_USE_LOADB

3161.4 GiB/s
#define TSM 32u                     // The tile-size in dimension M
#define TSN 64u                     // The tile-size in dimension N
#define TSK 4u                     // The tile-size in dimension K
#define WPTM 32u                     // The work-per-thread in dimension M
#define WPTN 2u                     // The work-per-thread in dimension N
#define WIDTHA 4u
#define WIDTHB 2u
//#define WONT_USE_LOADB

Transpose B:
336.65 GiB/s
#define TSM 32u                     // The tile-size in dimension M
#define TSN 64u                     // The tile-size in dimension N
#define TSK 4u                     // The tile-size in dimension K
#define WPTM 32u                     // The work-per-thread in dimension M
#define WPTN 2u                     // The work-per-thread in dimension N
#define WIDTHA 4u
#define WIDTHB 2u

24x1536 * 1536x6144:

24x48:
667.39 GiB/s
#define TSM 24u                     // The tile-size in dimension M
#define TSN 48u                     // The tile-size in dimension N
#define TSK 24u                     // The tile-size in dimension K
#define WPTM 2u                     // The work-per-thread in dimension M
#define WPTN 4u                     // The work-per-thread in dimension N


785.12 GiB/s
#define TSM 24u                     // The tile-size in dimension M
#define TSN 48u                     // The tile-size in dimension N
#define TSK 24u                     // The tile-size in dimension K
#define WPTM 8u                     // The work-per-thread in dimension M
#define WPTN 4u                     // The work-per-thread in dimension N


650.92 GiB/s
#define TSM 24u                     // The tile-size in dimension M
#define TSN 48u                     // The tile-size in dimension N
#define TSK 24u                     // The tile-size in dimension K
#define WPTM 24u                     // The work-per-thread in dimension M
#define WPTN 2u                     // The work-per-thread in dimension N
#define WIDTHB 2u

845.17 GiB/s
#define TSM 24u                     // The tile-size in dimension M
#define TSN 48u                     // The tile-size in dimension N
#define TSK 4u                     // The tile-size in dimension K
#define WPTM 24u                     // The work-per-thread in dimension M
#define WPTN 2u                     // The work-per-thread in dimension N
#define WIDTHB 2u

1162.9 GiB/s
#define TSM 24u                     // The tile-size in dimension M
#define TSN 48u                     // The tile-size in dimension N
#define TSK 8u                     // The tile-size in dimension K
#define WPTM 6u                     // The work-per-thread in dimension M
#define WPTN 6u                     // The work-per-thread in dimension N
#define WIDTHA 2u



24x24:
701.04 GiB/s
#define TSM 24u                     // The tile-size in dimension M
#define TSN 24u                     // The tile-size in dimension N
#define TSK 24u                     // The tile-size in dimension K
#define WPTM 2u                     // The work-per-thread in dimension M
#define WPTN 2u                     // The work-per-thread in dimension N


754.07 GiB/s 
#define TSM 24u                     // The tile-size in dimension M
#define TSN 24u                     // The tile-size in dimension N
#define TSK 24u                     // The tile-size in dimension K
#define WPTM 4u                     // The work-per-thread in dimension M
#define WPTN 4u                     // The work-per-thread in dimension N


815.90 GiB/s
#define TSM 24u                     // The tile-size in dimension M
#define TSN 24u                     // The tile-size in dimension N
#define TSK 12u                     // The tile-size in dimension K
#define WPTM 4u                     // The work-per-thread in dimension M
#define WPTN 4u                     // The work-per-thread in dimension N


867.35 GiB/s
#define TSM 24u                     // The tile-size in dimension M
#define TSN 24u                     // The tile-size in dimension N
#define TSK 6u                     // The tile-size in dimension K
#define WPTM 4u                     // The work-per-thread in dimension M
#define WPTN 4u                     // The work-per-thread in dimension N


959.66 GiB/s
#define TSM 24u                     // The tile-size in dimension M
#define TSN 24u                     // The tile-size in dimension N
#define TSK 8u                     // The tile-size in dimension K
#define WPTM 6u                     // The work-per-thread in dimension M
#define WPTN 4u                     // The work-per-thread in dimension N
