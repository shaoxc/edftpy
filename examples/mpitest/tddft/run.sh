if [ -z "$MPIRUN" ]
then
	mpirun=mpirun
else
	mpirun=$MPIRUN
fi
one=false
two=false
qe=false
clean=false
ptest=false
case $1 in
	"one")
		one=true ;;
	"two")
		two=true ;;
	"qe")
		qe=true ;;
	"clean")
		clean=true ;;
	"all")
		ptest=true ;
		one=true ;
		two=true ;
		qe=true ;;
	*)
		ptest=true ;;
esac

echo $ptest, $one, $two, $qe, $clean

if test "$ptest" = true; then
	$mpirun -n 4 python test_tddft.py | tee log.0
fi

if test "$one" = true; then
	$mpirun -n 4 python -m edftpy edftpy_1.ini | tee log.1
fi

if test "$qe" = true; then
	sed -i '/conv_thr/d' sub_ho.in
	sed -i '/dt/a nstep          =  10' sub_ho.in
	$mpirun -n 4 python run_qepy.py | tee log.qe
fi

if test "$two" = true; then
	$mpirun -n 4 python -m edftpy edftpy_2.ini | tee log.2
fi

if test "$clean" = true; then
	rm -r sub_* edftpy_gsystem.xyz edftpy_running.json log.*
fi
