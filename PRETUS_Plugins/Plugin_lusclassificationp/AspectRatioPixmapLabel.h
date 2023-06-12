#pragma once
#include <QLabel>
#include <QPixmap>
#include <QResizeEvent>
#include <QList>
#include <QStringList>

class AspectRatioPixmapLabel : public QLabel
{
    Q_OBJECT
public:

    static QList<QColor> color_scores;

    explicit AspectRatioPixmapLabel(QWidget *parent = 0);
    virtual int heightForWidth( int width ) const;
    virtual QSize sizeHint() const;
    QPixmap scaledPixmap() const;
    void setBlank_image_filename(const QString &value);

public Q_SLOTS:
    void setPixmap ( const QPixmap & );
    void setLabelPixmap ( const QPixmap & );
    void resizeEvent(QResizeEvent *);
    virtual void mouseMoveEvent ( QMouseEvent * event );
    virtual void mousePressEvent ( QMouseEvent * event );
    virtual void mouseReleaseEvent ( QMouseEvent * event );

    virtual void slot_selectZone(int i, int s);
    virtual void slot_submitZoneScore(int i, int s, int user_score, double user_confidence, int ai_score, double ai_confidence);
    virtual void slot_reset();

Q_SIGNALS:

    void signal_setZone(int, int); // args: first is zone (1 to 6), second is side (R=1, L=2)

private:
    QPixmap pix;
    /**
     * @brief contains the labels
     */
    QPixmap *labelPix;

    QString blank_image_filename;
    int mCurrentZone;
    int mCurrentSide;
};
