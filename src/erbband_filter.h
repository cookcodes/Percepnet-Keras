#include <stdio.h>

typedef struct{
    int begin;
    int end;
    float * weights;
}erb_weight_t;

float erb_weights_0[]={0.989992,6.12323e-17,};
float erb_weights_1[]={0.622359,0.768859,};
float erb_weights_2[]={0.911805,0.4509,};
float erb_weights_3[]={0.992252,0.199044,};
float erb_weights_4[]={0.988804,0.359535,};
float erb_weights_5[]={0.933132,0.659565,};
float erb_weights_6[]={0.751647,0.92569,0.165778,};
float erb_weights_7[]={0.378282,0.986163,0.716541,};
float erb_weights_8[]={0.697545,0.99734,0.631839,};
float erb_weights_9[]={0.072889,0.775099,0.996855,0.698233,0.0940211,};
float erb_weights_10[]={0.0792495,0.71587,0.99557,0.853042,0.399222,};
float erb_weights_11[]={0.521843,0.916855,0.985339,0.747048,0.306927,};
float erb_weights_12[]={0.170611,0.66477,0.951733,0.980541,0.773258,0.401812,};
float erb_weights_13[]={0.196314,0.634091,0.915722,0.999216,0.888586,0.622813,0.261376,};
float erb_weights_14[]={0.0395843,0.45871,0.78237,0.965237,0.991521,0.871392,0.634333,0.32134,};
float erb_weights_15[]={0.129944,0.490587,0.77306,0.946964,0.999745,0.934965,0.769011,0.52704,0.238848,};
float erb_weights_16[]={0.0225922,0.354739,0.639236,0.849841,0.971057,0.997894,0.934634,0.792979,0.589875,0.345324,0.0803551,};
float erb_weights_17[]={0.064873,0.35561,0.609249,0.807495,0.938483,0.996766,0.98279,0.902018,0.763853,0.580465,0.365651,0.133759,};
float erb_weights_18[]={0.184727,0.431699,0.645391,0.814285,0.930752,0.991014,0.994863,0.945222,0.847589,0.709437,0.539583,0.347608,0.1433,};
float erb_weights_19[]={0.101225,0.326428,0.530652,0.704769,0.841932,0.93764,0.989679,0.997962,0.964291,0.89207,0.785988,0.651693,0.495466,0.323918,0.143721,};
float erb_weights_20[]={0.0638084,0.264844,0.451896,0.618241,0.758482,0.868627,0.946085,0.989618,0.999253,0.976147,0.922441,0.841094,0.73571,0.610364,0.469439,0.317467,0.158985,};
float erb_weights_21[]={0.0386532,0.217115,0.386141,0.54089,0.677298,0.792122,0.882965,0.94827,0.987281,0.999999,0.98711,0.949909,0.890215,0.810277,0.71269,0.600294,0.476096,0.343186,0.204664,0.0635652,};
float erb_weights_22[]={0.00158342,0.160046,0.312526,0.455541,0.586047,0.701479,0.79978,0.879393,0.939268,0.978832,0.997978,0.997016,0.976646,0.937905,0.882126,0.810888,0.725969,0.629296,0.522904,0.408886,0.289357,0.166414,0.042105,};
float erb_weights_23[]={0.0771907,0.214856,0.346893,0.471014,0.585201,0.687728,0.777165,0.852391,0.912586,0.957221,0.986056,0.999113,0.996665,0.979209,0.947448,0.902263,0.844687,0.775885,0.69712,0.60974,0.515138,0.414749,0.310013,0.202359,0.0931932,};
float erb_weights_24[]={0.0816062,0.202854,0.319909,0.431186,0.53526,0.630875,0.716955,0.792602,0.857107,0.909936,0.950732,0.979311,0.995648,0.99987,0.992247,0.973177,0.943176,0.902862,0.852947,0.794216,0.72752,0.653758,0.57387,0.488821,0.399591,0.307157,0.212498,0.116566,0.0202997,};
float erb_weights_25[]={0.0161223,0.124284,0.230061,0.332296,0.429931,0.521999,0.607638,0.686088,0.756705,0.818948,0.872385,0.916694,0.95166,0.977162,0.993183,0.999794,0.997153,0.985497,0.965139,0.936457,0.899888,0.855922,0.805099,0.74799,0.6852,0.617367,0.545135,0.469167,0.390131,0.308696,0.225524,0.141271,0.0565739,};
float erb_weights_26[]={0.0754068,0.169689,0.261736,0.350783,0.436121,0.517103,0.59314,0.663709,0.728354,0.786675,0.838348,0.883109,0.920759,0.95116,0.974237,0.989971,0.998398,0.999609,0.993743,0.980984,0.961559,0.935736,0.903814,0.86613,0.823039,0.774932,0.722208,0.665296,0.604623,0.540641,0.473795,0.404545,0.333349,0.260654,0.186904,0.112551,0.038016,};
float erb_weights_27[]={0.0279468,0.111694,0.19409,0.274597,0.352701,0.427925,0.499819,0.567984,0.632044,0.691676,0.746579,0.796512,0.841253,0.880635,0.914518,0.942803,0.965432,0.982378,0.993646,0.999277,0.999342,0.99394,0.983199,0.967272,0.946333,0.920581,0.890236,0.855532,0.816719,0.774064,0.72784,0.678338,0.625852,0.570687,0.513141,0.453525,0.392159,0.329337,0.265381,0.200591,0.135265,0.0696981,0.00418192,};
float erb_weights_28[]={0.036276,0.109922,0.182537,0.253743,0.323193,0.390551,0.4555,0.51775,0.577036,0.633108,0.685747,0.73475,0.779942,0.821168,0.858305,0.891244,0.919898,0.944212,0.964144,0.979675,0.990809,0.997568,0.999991,0.998138,0.992083,0.981918,0.96775,0.949702,0.927901,0.9025,0.873652,0.841523,0.80629,0.768134,0.727247,0.683826,0.638068,0.59018,0.54037,0.488848,0.435824,0.381507,0.326111,0.269849,0.21292,0.155537,0.0978965,0.0402025,};
float erb_weights_29[]={0.0609994,0.125586,0.189306,0.251911,0.313157,0.372826,0.430689,0.486551,0.540221,0.591521,0.64029,0.686375,0.729645,0.76998,0.807271,0.841428,0.872369,0.900032,0.924366,0.945331,0.962903,0.97707,0.98783,0.995197,0.999192,0.999849,0.997214,0.991341,0.982293,0.970143,0.954973,0.936873,0.915938,0.892272,0.865983,0.837191,0.806012,0.772571,0.737001,0.69943,0.659997,0.618842,0.576099,0.53192,0.486436,0.439802,0.392154,0.343639,0.2944,0.244577,0.194318,0.143752,0.0930197,0.0422615,};
float erb_weights_30[]={0.0173583,0.0745896,0.131316,0.187352,0.242534,0.296692,0.349669,0.40132,0.451498,0.500073,0.546911,0.5919,0.634928,0.675892,0.714701,0.751268,0.785515,0.81738,0.846795,0.873716,0.898095,0.9199,0.939102,0.955682,0.96963,0.980939,0.989614,0.995664,0.999107,0.999965,0.998268,0.994052,0.987358,0.978233,0.96673,0.952904,0.936818,0.918539,0.898137,0.875684,0.851259,0.824943,0.796817,0.766968,0.73549,0.702467,0.667992,0.632161,0.595074,0.556817,0.517495,0.477204,0.436037,0.3941,0.351486,0.308293,0.264621,0.220562,0.176212,0.131671,0.0870238,0.0423624,};
float erb_weights_31[]={0.00840421,0.0588315,0.108908,0.158505,0.207508,0.2558,0.303273,0.349816,0.39533,0.439715,0.482884,0.524746,0.565216,0.604221,0.641685,0.677536,0.711717,0.744168,0.774837,0.803671,0.830635,0.855686,0.878793,0.899929,0.919068,0.936193,0.951291,0.964352,0.975373,0.984352,0.991293,0.996206,0.999102,0.999998,0.998912,0.995869,0.990896,0.984024,0.975285,0.964716,0.952357,0.938252,0.922443,0.904978,0.885908,0.865285,0.843162,0.819594,0.794639,0.768357,0.740809,0.712053,0.682155,0.651176,0.619184,0.586239,0.552412,0.517767,0.482371,0.44629,0.409593,0.372343,0.334611,0.29646,0.257957,0.219163,0.180153,0.140983,0.101716,0.0624158,0.0231501,};
float erb_weights_32[]={0.00221727,0.0466361,0.0907968,0.13463,0.178036,0.220951,0.263292,0.304986,0.345952,0.386134,0.425458,0.463861,0.50128,0.53766,0.572945,0.607082,0.640022,0.671716,0.702125,0.731208,0.758927,0.785246,0.810138,0.833571,0.855522,0.875967,0.894888,0.912268,0.928095,0.942356,0.955045,0.966156,0.975688,0.983639,0.990012,0.994814,0.99805,0.999732,0.999871,0.998483,0.995583,0.991191,0.985326,0.978014,0.969276,0.959139,0.947631,0.934783,0.920627,0.905191,0.888513,0.870628,0.851569,0.831377,0.810088,0.787745,0.764385,0.740052,0.714786,0.688631,0.661631,0.633827,0.605267,0.575992,0.546048,0.51548,0.484334,0.452656,0.420493,0.387885,0.354883,0.321529,0.287867,0.253944,0.219803,0.185496,0.151056,0.116526,0.081956,0.0473877,0.0128586,};
float erb_weights_33[]={0.0160301,0.0550597,0.0938847,0.132442,0.170681,0.20854,0.245977,0.282935,0.319366,0.355218,0.390444,0.425006,0.458851,0.491942,0.524242,0.555709,0.586308,0.616002,0.64476,0.67255,0.699344,0.725112,0.749829,0.773475,0.796023,0.817455,0.837754,0.856902,0.874883,0.891685,0.907296,0.921708,0.934911,0.9469,0.95767,0.967219,0.975544,0.982645,0.988525,0.993188,0.996636,0.998877,0.999917,0.999767,0.998435,0.995934,0.992276,0.987476,0.981547,0.974507,0.966372,0.957161,0.946893,0.935588,0.92327,0.909956,0.895674,0.880443,0.864289,0.847241,0.82932,0.810553,0.79097,0.770595,0.749458,0.727587,0.705007,0.681756,0.657859,0.633342,0.608242,0.582583,0.556401,0.529724,0.502586,0.47501,0.447036,0.418686,0.390002,0.361005,0.33173,0.302209,0.272474,0.242549,0.212466,0.182259,0.151954,0.121581,0.091169,0.0607524,0.0303498,};

erb_weight_t erbweights[]=
{

    {
        1,
        3,
        erb_weights_0,
    },    
    {
        1,
        3,
        erb_weights_1,
    },    
    {
        2,
        4,
        erb_weights_2,
    },    
    {
        3,
        5,
        erb_weights_3,
    },   
    {
        4,
        6,
        erb_weights_4,
    },    
    {
        5,
        7,
        erb_weights_5,
    },    
    {
        6,
        9,
        erb_weights_6,
    },    
    {
        7,
        10,
        erb_weights_7,
    },    
    {
        9,
        12,
        erb_weights_8,
    },    
    {
        10,
        15,
        erb_weights_9,
    },    
    {
        12,
        17,
        erb_weights_10,
    },    
    {
        15,
        20,
        erb_weights_11,
    },    
    {
        17,
        23,
        erb_weights_12,
    },    
    {
        20,
        27,
        erb_weights_13,
    },    
    {
        23,
        31,
        erb_weights_14,
    },    
    {
        27,
        36,
        erb_weights_15,
    },    
    {
        31,
        42,
        erb_weights_16,
    },    
    {
        36,
        48,
        erb_weights_17,
    },    
    {
        42,
        55,
        erb_weights_18,
    },    
    {
        48,
        63,
        erb_weights_19,
    },    
    {
        55,
        72,
        erb_weights_20,
    },    
    {
        63,
        83,
        erb_weights_21,
    },    
    {
        72,
        95,
        erb_weights_22,
    },    
    {
        83,
        108,
        erb_weights_23,
    },    
    {
        95,
        124,
        erb_weights_24,
    },    
    {
        108,
        141,
        erb_weights_25,
    },    
    {
        124,
        161,
        erb_weights_26,
    },    
    {
        141,
        184,
        erb_weights_27,
    },    
    {
        161,
        209,
        erb_weights_28,
    },    
    {
        184,
        238,
        erb_weights_29,
    },    
    {
        209,
        271,
        erb_weights_30,
    },    
    {
        238,
        309,
        erb_weights_31,
    },    
    {
        271,
        352,
        erb_weights_32,
    },    
    {
        309,
        400,
        erb_weights_33,
    },
};