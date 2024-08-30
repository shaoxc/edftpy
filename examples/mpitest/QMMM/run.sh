if [ -z "$MPIRUN" ]
then
	mpirun="mpirun -n 4"
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
    $mpirun python test_QMMM.py | tee log.0
fi

if test "$clean" = true; then
	rm -r sub_* edftpy_gsystem.xyz edftpy_running.json log.*
fi
