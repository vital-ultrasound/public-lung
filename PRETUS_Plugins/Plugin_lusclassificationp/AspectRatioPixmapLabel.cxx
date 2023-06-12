#include "AspectRatioPixmapLabel.h"
#include <iostream>
#include <QColor>

static QColor color_score0(70, 180, 80);
static QColor color_score1(50, 50, 20);
static QColor color_score2(100, 100, 20);
static QColor color_score3(200, 200, 20);
static QColor color_score4(250, 250, 20);
static QColor color_selected(50, 150, 255);
static QColor color_background(0, 0, 0);

QList<QColor> AspectRatioPixmapLabel::color_scores = {color_score0, color_score1, color_score2, color_score3, color_score4};

AspectRatioPixmapLabel::AspectRatioPixmapLabel(QWidget *parent) :
    QLabel(parent)
{
    labelPix = nullptr;
    mCurrentZone = -1;
    mCurrentSide = -1;
    this->setMinimumSize(1,1);
    setScaledContents(false);

    QObject::connect(this, &AspectRatioPixmapLabel::signal_setZone,
                     this, &AspectRatioPixmapLabel::slot_selectZone);

}

void AspectRatioPixmapLabel::setPixmap ( const QPixmap & p)
{
    pix = p;
    QLabel::setPixmap(scaledPixmap());
}

void AspectRatioPixmapLabel::setLabelPixmap ( const QPixmap & p)
{
    labelPix =  new QPixmap(p);
}

int AspectRatioPixmapLabel::heightForWidth( int width ) const
{
    return pix.isNull() ? this->height() : ((qreal)pix.height()*width)/pix.width();
}

QSize AspectRatioPixmapLabel::sizeHint() const
{
    int w = this->width();
    return QSize( w, heightForWidth(w) );
}

QPixmap AspectRatioPixmapLabel::scaledPixmap() const
{
    return pix.scaled(this->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

void AspectRatioPixmapLabel::resizeEvent(QResizeEvent * e)
{
    if(!pix.isNull())
        QLabel::setPixmap(scaledPixmap());
}

void AspectRatioPixmapLabel::mouseMoveEvent ( QMouseEvent * event )
{
    QLabel::mouseMoveEvent(event);
}

void AspectRatioPixmapLabel::mousePressEvent ( QMouseEvent * event )
{
    QLabel::mouseMoveEvent(event);
}

void AspectRatioPixmapLabel::mouseReleaseEvent ( QMouseEvent * event )
{
    QImage image(labelPix->scaled(this->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation).toImage());

    auto x = event->pos().x();
    auto y = event->pos().y();

    QRgb pixelval = image.pixel(x, y);
    QColor c(pixelval);

    auto val  =c.red();
    auto val2  =c.green();

    int zoneid = int(val/10); // 1 to 6
    int side_id = int(val2/100); // 1 ==  left, 2==right

    // emit a signal qith the selected value
    Q_EMIT this->signal_setZone(zoneid, side_id);

}

void AspectRatioPixmapLabel::slot_reset(){

    QImage image_labels(labelPix->scaled(this->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation).toImage());
    QImage image_displaid(pix.scaled(this->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation).toImage());

    for (int ii=0; ii<image_labels.width(); ii++){
        for (int jj=0; jj<image_labels.height(); jj++){
            QColor c(image_labels.pixel(ii,jj));
            //QColor current_c(image_displaid.pixel(ii,jj));
            if (c != color_background){
                image_displaid.setPixelColor(ii, jj, color_background);
            }
        }
    }

    this->setPixmap(QPixmap::fromImage(image_displaid));

}

void AspectRatioPixmapLabel::slot_submitZoneScore(int i, int s, int user_score, double user_confidence, int ai_score, double ai_confidence){
    if (i <= 0 || s<=0 || user_score < 0){
        return;
    }

    QImage image_labels(labelPix->scaled(this->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation).toImage());
    QImage image_displaid(pix.scaled(this->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation).toImage());

    if (i == mCurrentZone && s == mCurrentSide){
        for (int ii=0; ii<image_labels.width(); ii++){
            for (int jj=0; jj<image_labels.height(); jj++){
                QColor c(image_labels.pixel(ii,jj));
                QColor current_c(image_displaid.pixel(ii,jj));
                if (current_c==color_selected){
                    image_displaid.setPixelColor(ii, jj, color_scores[user_score]);
                }
            }
        }
        mCurrentZone = -1;
        mCurrentSide = -1;
    }
    this->setPixmap(QPixmap::fromImage(image_displaid));

}

void AspectRatioPixmapLabel::slot_selectZone(int i, int s){
    if (i <= 0){
        // zone id 0 uis backgorund
        return;
    }

    QImage image_labels(labelPix->scaled(this->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation).toImage());
    QImage image_displaid(pix.scaled(this->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation).toImage());

    if (i == mCurrentZone && s == mCurrentSide){
        for (int ii=0; ii<image_labels.width(); ii++){
            for (int jj=0; jj<image_labels.height(); jj++){
                QColor c(image_labels.pixel(ii,jj));
                QColor current_c(image_displaid.pixel(ii,jj));
                if (c.red() == i*10 && c.green()==s*100 && current_c==color_selected){
                    image_displaid.setPixelColor(ii, jj, color_background);
                }
            }
        }
        mCurrentZone = -1;
        mCurrentSide = -1;
    } else {
        for (int ii=0; ii<image_labels.width(); ii++){
            for (int jj=0; jj<image_labels.height(); jj++){
                QColor c(image_labels.pixel(ii,jj));
                if (c.red() == i*10 && c.green()==s*100){
                    image_displaid.setPixelColor(ii, jj, color_selected);
                } else {
                    QColor current_c(image_displaid.pixel(ii,jj));
                    if (current_c==color_selected) {
                        // if a previous zone was selected, unselect it.
                        image_displaid.setPixelColor(ii, jj, color_background);
                    }
                }
            }
        }

        mCurrentZone = i;
        mCurrentSide = s;
    }
    this->setPixmap(QPixmap::fromImage(image_displaid));
}

void AspectRatioPixmapLabel::setBlank_image_filename(const QString &value)
{
    blank_image_filename = value;
}
