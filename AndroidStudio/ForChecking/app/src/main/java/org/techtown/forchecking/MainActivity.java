package org.techtown.forchecking;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.app.ActivityManager;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.ramotion.circlemenu.CircleMenuView;

public class MainActivity extends AppCompatActivity {
    Button start_stop;
    CircleMenuView circleMenu;
    AlertDialog alert;
    boolean started = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // FloatingWindow에서 가져온 좌표
        Intent coord_intent = getIntent();
        //FloatingWindow에 보낼
        Intent myIntent = new Intent(this, FloatingWindow.class);

        // start_stop : 버튼, circleMenu : 원으로된 메뉴
        start_stop = findViewById(R.id.start_stop);
        circleMenu = findViewById(R.id.circleMenu);

//        int coord_x = coord_intent.getExtras().getInt("coord_x");
//        int coord_y = coord_intent.getExtras().getInt("coord_y");

        start_stop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                start_stop();

                if (circleMenu.getVisibility() == View.INVISIBLE) {
                    circleMenu.setVisibility(View.VISIBLE);
                }
                else{
                    circleMenu.setVisibility(View.INVISIBLE);
                }
            }
        });

        circleMenu.setEventListener(new CircleMenuView.EventListener(){
            @Override
            public void onButtonClickAnimationStart(@NonNull CircleMenuView view, int buttonIndex) {
                super.onButtonClickAnimationStart(view, buttonIndex);
                switch (buttonIndex){
                    case 0:
                        Toast.makeText(MainActivity.this, "페이지 넘기기로 설정되었습니다.", Toast.LENGTH_SHORT).show();
                        break;

                    case 1:
                        Toast.makeText(MainActivity.this, "스크롤로 설정되었습니다.", Toast.LENGTH_SHORT).show();
                        // 전체 화면을 뭐라할까.. 밑에처럼하면 전체 화면이 스크롤되는게 아니라 버튼의 위치가 바뀜.
                        view.scrollBy(0,100);
                        break;
                }
            }
        });

        if(isMyServieRunning(org.techtown.forchecking.FloatingWindow.class)){
            started = true;
            start_stop.setText("Stop");
        }
    }

    public void start_stop() {
        if (checkPermission()) {
            if (started) {
                stopService(new Intent(MainActivity.this, org.techtown.forchecking.FloatingWindow.class));
                start_stop.setText("Start");
                started = false;
            } else {
                startService(new Intent(MainActivity.this, org.techtown.forchecking.FloatingWindow.class));
                start_stop.setText("Stop");
                started = true;
            }
        } else {
            reqPermission();
        }
    }

    private boolean checkPermission() {
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.M) {
            if (!Settings.canDrawOverlays(this)) {
                return false;
            } else {
                return true;
            }
        } else {
            return true;
        }
    }

    private void reqPermission(){
        final AlertDialog.Builder alertBuilder = new AlertDialog.Builder(this);
        alertBuilder.setCancelable(true);
        alertBuilder.setTitle("Screen overlay detected");
        alertBuilder.setMessage("Enable 'Draw over other apps' in your system setting.");
        alertBuilder.setPositiveButton("OPEN SETTINGS", new DialogInterface.OnClickListener(){
            @Override
            public void onClick(DialogInterface dialog, int which){
                Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:" + getPackageName()));
                startActivityForResult(intent, RESULT_OK);
            }
        });

        alert = alertBuilder.create();
        alert.show();
    }

    private boolean isMyServieRunning(Class<?> serivceClass){
        ActivityManager manager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        for(ActivityManager.RunningServiceInfo service : manager.getRunningServices(Integer.MAX_VALUE)){
            if(serivceClass.getName().equals(service.service.getClassName())){
                return true;
            }
        }
        return false;
    }
}