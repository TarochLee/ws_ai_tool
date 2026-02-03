#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Vision/Vision.h>

#include "ws_ai/ocr_vision.h"

#include <string>

namespace ws_ai {

static CGImageRef load_cgimage(const std::string& path_utf8) {
    NSString* p = [NSString stringWithUTF8String:path_utf8.c_str()];
    NSURL* url = [NSURL fileURLWithPath:p];
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
    if (!src) return nil;
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
    CFRelease(src);
    return img;
}

std::string ocr_with_vision(const std::string& image_path) {
    CGImageRef img = load_cgimage(image_path);
    if (!img) return {};

    __block NSMutableString* acc = [NSMutableString string];

    VNRecognizeTextRequest* req = [[VNRecognizeTextRequest alloc]
        initWithCompletionHandler:^(VNRequest* request, NSError* error) {
            if (error) return;
            NSArray<VNRecognizedTextObservation*>* results =
                (NSArray<VNRecognizedTextObservation*>*)request.results;
            if (![results isKindOfClass:[NSArray class]]) return;

            for (VNRecognizedTextObservation* obs in results) {
                VNRecognizedText* top = [[obs topCandidates:1] firstObject];
                if (top && top.string.length > 0) {
                    [acc appendString:top.string];
                    [acc appendString:@"\n"];
                }
            }
        }];

    // 中文优先 + 英文兜底
    req.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
    req.usesLanguageCorrection = YES;
    req.minimumTextHeight = 0.012;
    req.recognitionLanguages = @[ @"zh-Hans", @"en-US" ];

    VNImageRequestHandler* handler =
        [[VNImageRequestHandler alloc] initWithCGImage:img options:@{}];
    NSError* err = nil;
    [handler performRequests:@[ req ] error:&err];

    CGImageRelease(img);
    if (err) return {};

    const char* c = [acc UTF8String];
    return c ? std::string(c) : std::string();
}

} // namespace ws_ai