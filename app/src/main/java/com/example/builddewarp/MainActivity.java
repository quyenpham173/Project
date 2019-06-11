package com.example.builddewarp;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.RelativeLayout;
import com.example.quyenpham.R;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import static com.example.quyenpham.R.id.activity_main;

public class MainActivity extends AppCompatActivity {
    private TextToSpeech textToSpeech;
    final String xin_chao = "Xin chào, vui lòng chạm vào màn hình điện thoại để chụp ảnh";
    private static final String TAG = "MainActivity1";
    private static final int requestCode = 100;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.d(TAG, "onCreate: ");
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        RelativeLayout mh = (RelativeLayout) findViewById(activity_main);
        mh.setBackgroundResource(R.drawable.anh2);
        checkPermissions();
        final Locale loc = new Locale("vi");
        Log.d("Locale Available: ", Arrays.toString(loc.getAvailableLocales()));
        textToSpeech = new TextToSpeech(MainActivity.this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != TextToSpeech.ERROR){
                    textToSpeech.setLanguage(loc);
                    textToSpeech.speak(xin_chao, TextToSpeech.QUEUE_FLUSH,null, null);
                }
            }
        });
        RelativeLayout relativeLayout = (RelativeLayout) findViewById(activity_main);
        relativeLayout.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                textToSpeech.stop();
                Intent intent = new Intent(MainActivity.this, CaptureImage.class);
                startActivity(intent);

            }
        });
    }

    String[] permissions = new String[]{
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.CAMERA,
    };

    private void checkPermissions() {
        int result;
        List<String> listPermissionsNeeded = new ArrayList<>();
        for (String p : permissions) {
            result = ContextCompat.checkSelfPermission(this, p);
            if (result != PackageManager.PERMISSION_GRANTED) {
                listPermissionsNeeded.add(p);
            }
        }
        if (!listPermissionsNeeded.isEmpty()) {
            ActivityCompat.requestPermissions(this, listPermissionsNeeded.toArray(new String[listPermissionsNeeded.size()]), 100);
        }
    }
}
