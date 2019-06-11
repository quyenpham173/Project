package com.example.builddewarp;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Environment;
import android.util.Log;
import android.widget.TextView;

import com.googlecode.tesseract.android.TessBaseAPI;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class ImageToText extends AsyncTask<Bitmap,String,String> {
    //private static final String
    private static final String TAG = MainActivity.class.getSimpleName();
    public static final String TESS_DATA = "/tessdata";
    private static final String DATA_PATH = Environment.getExternalStorageDirectory().toString() + "/Tess1";
    private Context contextImageToText;
    private TextView textView;
    private String languge="vie";

    ImageToText(Context context){
        contextImageToText=context;
        //  textView=t;
    }

    @Override
    protected void onPreExecute(){
        super.onPreExecute();
        //  textView.setText("init");
    }


    @Override
    protected String doInBackground(Bitmap... bitmaps) {
        //prepareTessData();
        publishProgress("init process");
        TessBaseAPI tessBaseAPI = new TessBaseAPI();
        tessBaseAPI.init(DATA_PATH,languge);
        publishProgress("processing");
        tessBaseAPI.setImage(bitmaps[0]);
        return tessBaseAPI.getUTF8Text();
    }

    @Override
    protected void onProgressUpdate(String... s){
        super.onProgressUpdate(s[0]);
        //   textView.setText(s[0]);
    }

    @Override
    protected void onPostExecute(String s) {
        super.onPostExecute(s);
        //    textView.setText(s);

    }

    void prepareTessData(){
        AssetManager assetManager=contextImageToText.getAssets();
        InputStream in=null;
        OutputStream out=null;
        try{
            File dir = new File(DATA_PATH + TESS_DATA);
            if(!dir.exists()){
                dir.mkdirs();
            }
            String pathToDataFile = DATA_PATH+TESS_DATA+"/"+languge+".traineddata";
            if(!(new File(pathToDataFile)).exists()){
                in = assetManager.open("tessdata/"+languge+".traineddata");
                out = new FileOutputStream(pathToDataFile);
                byte [] buff = new byte[1024];
                int len ;
                while(( len = in.read(buff)) > 0){
                    out.write(buff,0,len);
                }
            }
            // }
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
        }
        finally {
            if(in!=null) {
                try {
                    in.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if(out!=null){
                try {
                    out.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }


}

