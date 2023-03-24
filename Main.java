import org.opencv.core.*;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.SIFT;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import static org.opencv.imgcodecs.Imgcodecs.*;

class Main{
    public static void main(String[]args){
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        //specifying the path of incoming images
        File cameraFile = new File(System.getProperty("user.dir") +"\\src\\input\\Camera");
        File mobileFile = new File(System.getProperty("user.dir") +"\\src\\input\\Mobile");




        //fetching Images
        List <Mat> cameraImgs = new ArrayList<>(List.of());
        List <Mat> mobileImgs = new ArrayList<>(List.of());

        System.out.println("Going To Camera read files");
        Arrays.stream(Objects.requireNonNull(cameraFile.listFiles())).toList()
                .forEach(img-> cameraImgs.add(Imgcodecs.imread(img.toString(),IMREAD_GRAYSCALE)));
        System.out.println("Going To read Mobile files");
        Arrays.stream(Objects.requireNonNull(mobileFile.listFiles())).toList()
                .forEach(img-> mobileImgs.add(Imgcodecs.imread(img.toString(), Imgcodecs.IMREAD_REDUCED_GRAYSCALE_2)));


//        System.out.println(mobileImgs.size());
//        for (int i=0;i<5;i++) {
//            int finalI = i;
//            Imgcodecs.imwrite(System.getProperty("user.dir") +"\\src\\output\\lol\\"+ finalI + ".JPG",mobileImgs.get(finalI));
//            }


//        Detecting Features at first
//        Images from the canon Camera
        SIFT sift = SIFT.create();
        ArrayList<MatOfKeyPoint> matOfKeyPointCamera = new ArrayList<>();
        ArrayList<MatOfKeyPoint> matOfKeyPointMobile = new ArrayList<>();

        ArrayList<Mat> matOfDescriptorsCamera = new ArrayList<>();
        ArrayList<Mat> matOfDescriptorsMobile = new ArrayList<>();

        //Computing extraction of keyPoints
        System.out.println("Going To Detect key points");

        cameraImgs.forEach(img-> {
            MatOfKeyPoint matOfKeyPoint = new MatOfKeyPoint();
            Mat descriptor = new Mat();
            sift.detectAndCompute(img,new Mat(),matOfKeyPoint,descriptor,false);
            matOfKeyPointCamera.add(matOfKeyPoint);
            matOfDescriptorsMobile.add(descriptor);
           System.out.println("key points detected");
        });

        int i = 0;
        mobileImgs.forEach( img-> {
            MatOfKeyPoint matOfKeyPoint = new MatOfKeyPoint();
            Mat descriptor = new Mat();
            sift.detectAndCompute(img, new Mat(),matOfKeyPoint,descriptor,false);
            matOfKeyPointMobile.add(matOfKeyPoint);
            matOfDescriptorsMobile.add(descriptor);
            System.out.println("key points detected");
        });

        BFMatcher matcher = BFMatcher.create();
        List <MatOfDMatch> matches = new ArrayList<>(List.of());
        for (i = 0;i < matOfDescriptorsMobile.size();i++) {
            for (int j = i; j <matOfDescriptorsMobile.size();j++) {
                MatOfDMatch match = new MatOfDMatch();
                matcher.match(matOfDescriptorsMobile.get(i), matOfDescriptorsMobile.get(j), match, new Mat());
                matches.add(match);
                Imgcodecs.imwrite(System.getProperty("user.dir") + "\\src\\output\\mobile\\" + i * 500 + j + ".JPG", match);
                System.out.println(match);
            }
        }

        i = 0;
        System.out.println(matches.size() + " size of matches");
        for (MatOfDMatch mat: matches) {
            System.out.println(mat);
            Imgcodecs.imwrite(System.getProperty("user.dir") +"\\src\\output\\mobile\\" + i+5 + ".JPG", mat);
            i++;
        }
        //Doing the matches for camera images and computing them
        matches.clear();
        for (i = 0;i < matOfDescriptorsCamera.size();i++) {
            for (int j = i+1; j < matOfDescriptorsCamera.size();j++) {

                MatOfDMatch match = new MatOfDMatch();
                matcher.match(matOfDescriptorsCamera.get(i),matOfDescriptorsCamera.get(j),match,new Mat());
                matches.add(match);
                Imgcodecs.imwrite(System.getProperty("user.dir") +"\\src\\output\\camera\\" + i*500+j + ".JPG", match);
                System.out.println(match);
            }
        }

             //Writing images with features detected marked
        System.out.println("Going to write on files");
            for (i = 0; i < matOfKeyPointCamera.size(); i++) {
                int finalI = i;
                matOfKeyPointCamera.get(finalI).toList().forEach(keyPoint -> {
                    Point point = new Point(keyPoint.pt.x, keyPoint.pt.y);
                    Imgproc.drawMarker(cameraImgs.get(finalI), point, new Scalar(0, 0, 0), Imgproc.MARKER_SQUARE, 4, 4, Imgproc.LINE_8);
                });
                System.out.println("Going to write on files");
                Imgcodecs.imwrite(System.getProperty("user.dir") +"\\src\\output\\camera\\" + i + ".JPG", cameraImgs.get(finalI));
            }

        System.out.println("Going to write on files");
            for (i = 0; i < matOfKeyPointMobile.size(); i++) {
                int finalI = i;
                matOfKeyPointMobile.get(finalI).toList().forEach(keyPoint -> {
                    Point point = new Point(keyPoint.pt.x, keyPoint.pt.y);
                    Imgproc.drawMarker(mobileImgs.get(finalI), point, new Scalar(0, 0, 0), Imgproc.MARKER_SQUARE, 4, 4, Imgproc.LINE_8);
                });
                System.out.println("Going to write on files");
                Imgcodecs.imwrite(System.getProperty("user.dir") +"\\src\\output\\mobile\\" + i + ".JPG", mobileImgs.get(finalI));
            }

    }
}