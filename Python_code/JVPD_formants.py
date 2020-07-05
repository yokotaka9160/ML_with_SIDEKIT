numfile = 385;
for i = 1:numfile
wav = cellstr(FileName1(i,1))
filename = char(strcat(wav,'.wav'));
filename
[mtlb,Fs] = audioread(filename);
segmentlen = 100;
noverlap = 90;
NFFT = 128;

spectrogram(mtlb,segmentlen,noverlap,NFFT,Fs,'yaxis')
title('Signal Spectrogram')

dt = 1/Fs;
I0 = round(0.1/dt);
Iend = round(0.25/dt);
x = mtlb(I0:Iend);

x1 = x.*hamming(length(x));

preemph = [1 0.63];
x1 = filter(1,preemph,x1);

A = lpc(x1,8);
rts = roots(A);

rts = rts(imag(rts)>=0);
angz = atan2(imag(rts),real(rts));

[frqs,indices] = sort(angz.*(Fs/(2*pi)));
bw = -1/2*(Fs/(2*pi))*log(abs(rts(indices)));

nn = 1;
for kk = 1:length(frqs)
    if (frqs(kk) > 500 && bw(kk) <3000)
        formants(nn) = frqs(kk);
        formants_list2(i,nn) = formants(nn)
        nn = nn+1;
    end
formants
end
end