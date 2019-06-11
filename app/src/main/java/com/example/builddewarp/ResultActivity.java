package com.example.builddewarp;

import android.content.Intent;
import android.os.Bundle;
import android.os.Environment;
import android.speech.tts.TextToSpeech;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.RelativeLayout;
import android.widget.Toast;

import com.example.quyenpham.R;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Locale;
import java.util.UUID;

import static com.example.quyenpham.R.id.activity_result;

public class ResultActivity extends AppCompatActivity {
    public static final String ROOT_FOLDER = "Reading Assistance";
    public static final String FILE_TXT = "Page_";
    private TextToSpeech textToSpeech;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);
        RelativeLayout mh = (RelativeLayout) findViewById(activity_result);
        mh.setBackgroundResource(R.drawable.anh2);
        Bundle bundle = getIntent().getExtras();

        assert bundle != null;
        final String content = bundle.getString("RESULT");
        assert content != null;
        final CharSequence charSequence = new StringBuffer(content);
        Log.d("Quyen", content);
        FileOutputStream fos = null;
        File textFile = createTextFile();
        try {
            fos = new FileOutputStream(textFile);
            fos.write(content.getBytes());
            Toast.makeText(ResultActivity.this, "save to " + getFilesDir() +
                            "/" + FILE_TXT,
                    Toast.LENGTH_SHORT).show();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            if (fos!=null){
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        final Locale loc = new Locale("vi");
        textToSpeech = new TextToSpeech(ResultActivity.this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != TextToSpeech.ERROR){
                    textToSpeech.setLanguage(loc);
                    String utteranceId = UUID.randomUUID().toString();
                    textToSpeech.speak(charSequence, TextToSpeech.QUEUE_FLUSH,null, utteranceId);
                }
            }
        });

        RelativeLayout relativeLayout = (RelativeLayout) findViewById(activity_result);
        relativeLayout.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                textToSpeech.stop();
                Intent intent = new Intent(ResultActivity.this, CaptureImage.class);
                startActivity(intent);
            }
        });
    }
    private File createTextFile() {
        String dirPath = Environment.getExternalStorageDirectory().getAbsolutePath() +
                File.separator + ROOT_FOLDER;
        File dir = new File(dirPath);
        if (!dir.exists())
            dir.mkdirs();
        String fileName = FILE_TXT + System.currentTimeMillis() + ".txt";
        return new File(dir, fileName);
    }

    @Override
    protected void onPause() {
        if (textToSpeech != null){
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        super.onPause();
    }
}

