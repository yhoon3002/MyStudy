package org.techtown.forchecking;

import android.app.Service;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.os.IBinder;
import android.os.Parcelable;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.ActivityChooserView;
import androidx.constraintlayout.widget.ConstraintLayout;

import com.ramotion.circlemenu.CircleMenuView;

import org.techtown.forchecking.MainActivity;
import org.techtown.forchecking.R;

public class FloatingWindow extends Service {

    WindowManager wm;
    LinearLayout ll;

    @Nullable
    @Override
    public IBinder onBind(Intent intent){
        return null;
    }

    @Override
    public void onCreate(){
        super.onCreate();;

        // MainActivity로 좌표를 보내기 위해 사용
        Intent coord_intent = new Intent(this, MainActivity.class);
        // MainActivity로 ImageView를 보내기 위해 사용
        Intent img_intent = new Intent(this, MainActivity.class);

        wm = (WindowManager) getSystemService(WINDOW_SERVICE);

        ll = new LinearLayout(this);
        ll.setBackgroundColor(Color.TRANSPARENT);
        LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT);
        ll.setLayoutParams(layoutParams);

        final WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                PixelFormat.TRANSLUCENT);

        params.gravity = Gravity.CENTER;
        params.x = 0;
        params.y = 0;

        ImageView openapp = new ImageView(this);
        openapp.setImageResource(R.mipmap.ic_launcher_round);;
        ViewGroup.LayoutParams butnparams = new ViewGroup.LayoutParams(100, 100);
        openapp.setLayoutParams(butnparams);
        // openapp을 mainactivity로 넘겨주려면..?
//        img_intent.putExtra("image", openapp);

        ll.addView(openapp);
        wm.addView(ll,params);

        // openapp을 눌렀을 때, 이동시켰을 때의 처리
        openapp.setOnTouchListener(new View.OnTouchListener(){
            WindowManager.LayoutParams updatepar = params;
            double x, y, px, py;

            @Override
            public boolean onTouch(View view, MotionEvent motionEvent){
                switch(motionEvent.getAction()){
                    case MotionEvent.ACTION_DOWN:
                        x = updatepar.x;
                        y = updatepar.y;

                        px = motionEvent.getRawX();
                        py = motionEvent.getRawY();

                        break;

                    case MotionEvent.ACTION_MOVE:
                        updatepar.x = (int) (x+(motionEvent.getRawX()-px));
                        updatepar.y = (int) (y+(motionEvent.getRawY()-py));

                        //좌표 MainActivity로 보내기
//                        coord_intent.putExtra("coord_x",updatepar.x);
//                        coord_intent.putExtra("coord_y",updatepar.y);

                        wm.updateViewLayout(ll, updatepar);

                    default:
                        break;
                }

                return false;
            }
        });

        openapp.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){

            }
        });
    }

    @Override
    public void onDestroy(){
        super.onDestroy();
        stopSelf();
        wm.removeView(ll);
    }
}
