package com.example.builddewarp;

import android.content.Intent;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.RelativeLayout;
import android.widget.TextView;

import com.example.tesseract.R;

import java.util.Locale;

import static com.example.tesseract.R.id.activity_main;

public class MainActivity extends AppCompatActivity {
    private TextToSpeech textToSpeech;
    final String contentWelcome = "Hello, Welcome to reading assitance application, please press screen to take photo";
    private RelativeLayout mh;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mh = (RelativeLayout) findViewById(activity_main);
//        mh.setBackgroundResource(R.drawable.anh2);

        textToSpeech = new TextToSpeech(MainActivity.this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != TextToSpeech.ERROR){
                    textToSpeech.setLanguage(Locale.US);
                    textToSpeech.speak(contentWelcome, TextToSpeech.QUEUE_FLUSH,null);
                }
            }
        });
        RelativeLayout relativeLayout = (RelativeLayout) findViewById(activity_main);
        relativeLayout.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, CaptureImage.class);
                startActivity(intent);

            }
        });
    }
}
