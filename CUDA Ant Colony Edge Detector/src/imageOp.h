void showImage(IplImage *img){
	cvNamedWindow("Slika", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("Slika", 440, 65);
	cvShowImage("Slika", img);
	while(true) if (cvWaitKey(10) == 27) break;
	cvDestroyWindow("Slika");
}
