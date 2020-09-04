my $in = shift;
my $out = shift;
open F1, "$in";
open F2, ">$out";

my %hash;
while(<F1>){
	chomp;
	my @arr = split(/\t/, $_);
	my @arr_d = split(/\,/, $arr[0]);

	my $len = @arr_d;

	my $mark = 0;
	for(my $i=0; $i<$len; $i++){
	    if($arr_d[$i] == 1){
	        $mark = 1;
	        last;                       #1528
	    }else{

	    }
	}


    if($mark == 1){
        if(exists $hash{$arr[0]}){
            $hash{$arr[0]} = "remove";          #2次／4个
        }else{
            $hash{$arr[0]} = $arr[1];           #1522个
        }
    }elsif($mark == 0 && $arr[1] == "1,0"){
        $hash{$arr[0]} = $arr[1];                           #1次／2个
    }

}

foreach my $key (keys %hash){
    if($hash{$key} eq "remove"){

    }else{
        print F2 "$key\t$hash{$key}\n";
    }
}
